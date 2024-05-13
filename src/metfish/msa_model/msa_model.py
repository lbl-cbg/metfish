
import os
import time
import torch
import logging
import pandas as pd
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader

from openfold.data.data_transforms import make_atom14_masks
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.np import residue_constants
from openfold.utils.validation_metrics import (drmsd, gdt_ts, gdt_ha,)
from openfold.utils.loss import lddt_ca, AlphaFoldLoss
from openfold.utils.superimposition import superimpose
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.tensor_utils import tensor_tree_map

from metfish.msa_model.config import model_config
from metfish.msa_model.data.data_pipeline import DataPipeline
from metfish.msa_model.data.feature_pipeline import FeaturePipeline
from metfish.msa_model.model.alphafold_saxs import AlphaFoldSAXS


# create dataset
class MSASAXSDataset(Dataset): 

  def __init__(self, config, path, data_dir=None, pdb_dir=None, msa_dir=None, saxs_dir=None):
      self.pdb_chains = pd.read_csv(path, index_col='name')#.sort_index()
      self.data_dir = data_dir
      self.msa_dir = msa_dir
      self.pdb_dir = pdb_dir
      self.saxs_dir = saxs_dir
      self.data_pipeline = DataPipeline(template_featurizer=None)
      self.feature_pipeline = FeaturePipeline(config) 
      
  def __len__(self):
      return len(self.pdb_chains)
  
  def __getitem__(self, idx):
      item = self.pdb_chains.iloc[idx]
      
      # sequence data
      sequence_feats = self.data_pipeline.process_str(item.seqres, item.name)
      
      # msa data
      msa_id = item.msa_id if hasattr(item, 'msa_id') else item.name
      msa_features = self.data_pipeline._process_msa_feats(f'{self.msa_dir}/{msa_id}', item.seqres, alignment_index=None)
      # NOTE - could also manipulate the clustering process here

      # saxs data
      saxs_features = self.data_pipeline._process_saxs_feats(f'{self.saxs_dir}/{item.name}.pdb.pr.csv')

      # pdb data
      pdb_features = self.data_pipeline.process_pdb_feats(f'{self.pdb_dir}/fixed_{item.name}.pdb')
      data = {**sequence_feats, **msa_features, **saxs_features, **pdb_features}

      feats = self.feature_pipeline.process_features(data)

      feats["batch_idx"] = torch.tensor(
            [idx for _ in range(feats["aatype"].shape[-1])],
            dtype=torch.int64,
            device=feats["aatype"].device) 
              
      return feats

# define the lightning module for training
class MSASAXSModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = AlphaFoldSAXS(config)
        # self.model = MSASAXSModule(config)
        self.loss = AlphaFoldLoss(config.loss)
        self.ema = ExponentialMovingAverage(
                model=self.model, decay=config.ema.decay
            )
        self.cached_weights = None
        self.last_lr_step = -1

        self.last_log_time = time.time()

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True,
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                )

        with torch.no_grad():
            metrics = self._compute_validation_metrics(
                batch, 
                outputs,
                superimposition_metrics=(not train)
            )

        for k,v in metrics.items():
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                on_step=False, on_epoch=True, logger=True
            )
        
        self.log('dur', time.time() - self.last_log_time)
        self.last_log_time = time.time()

    def training_step(self, batch):
        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)
                
        outputs = self.model(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        loss, loss_breakdown = self.loss(outputs, batch, _return_breakdown=True)

        self._log(loss_breakdown, batch, outputs)

        return loss
    
    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])
        
        # Run the model
        outputs = self.model(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        batch["use_clamped_fape"] = 0.

        # Compute loss and other metrics
        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False)        

    def on_validation_epoch_end(self):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        
        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
    
        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]
    
        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=1e-6,
            per_residue=False,
        )
   
        metrics["lddt_ca"] = lddt_ca_score
   
        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
   
        metrics["drmsd_ca"] = drmsd_ca_score
    
        if(superimposition_metrics):
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score
    
        return metrics
    
    def configure_optimizers(self, learning_rate: float = 1e-3, eps: float = 1e-5,):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, eps=eps
            )
        
        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = learning_rate

        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            last_epoch=self.last_lr_step
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }    

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        if(not self.model.template_config.enabled):
            ema["params"] = {k:v for k,v in ema["params"].items() if not "template" in k}
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step
        
    def load_from_jax(self, jax_path):
        import_jax_weights_(
                self.model, jax_path, version='model_3'
        )


if __name__ == "__main__":
    # set up data paths and configuration
    ckpt_path = None
    metfish_dir = "/global/cfs/cdirs/m3513/metfish"
    data_dir = f"{metfish_dir}/PDB70_verB_fixed_data/result"
    msa_dir = f"{metfish_dir}/PDB70_verB_fixed_data/result_subset/"
    training_csv = f'{msa_dir}/input_training.csv'  # was input.csv in apo_holo_data
    val_csv = f'{msa_dir}/input_validation.csv'
    pdb_dir = f"{data_dir}/pdb"
    saxs_dir = f"{data_dir}/saxs_r"
    os.environ["MODEL_DIR"] = "/pscratch/sd/s/smprince/projects/alphaflow/src/alphaflow/working_dir"

    # ckpt_path = "/pscratch/sd/s/smprince/projects/openfold/openfold/resources/openfold_params/initial_training.pt"
    # ckpt_path = "/global/cfs/cdirs/m3513/metfish/alphaflow_weights/alphaflow_pdb_base_202402.pt" # TODO - switch alphaflow checkpoint for openfold version
    ckpt_path = None
    jax_param_path = "/pscratch/sd/s/smprince/projects/alphaflow/params_model_1.npz"
    resume_model_weights_only = False
    deterministic = False

    config = model_config('initial_training', train=True, low_prec=True) 
    if deterministic:
        config.data.eval.masked_msa_replace_fraction = 0.0
        config.model.global_config.deterministic = True
    data_config = config.data
    data_config.common.use_templates = False
    data_config.common.max_recycling_iters = 0

    # set up training and test datasets and dataloaders
    train_dataset = MSASAXSDataset(data_config, training_csv, msa_dir=msa_dir, saxs_dir=saxs_dir, pdb_dir=pdb_dir)
    val_dataset = MSASAXSDataset(data_config, val_csv, msa_dir=msa_dir, saxs_dir=saxs_dir, pdb_dir=pdb_dir)
    train_dataset[0]

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # initialize model and trainer
    msasaxsmodel = MSASAXSModel(config)
    trainer = pl.Trainer(accelerator="gpu", 
                        max_epochs=100, 
                        gradient_clip_val=1.,
                        limit_train_batches=1.0, 
                        limit_val_batches=1.0,
                        callbacks=[ModelCheckpoint(
                                dirpath=os.environ["MODEL_DIR"], 
                                save_top_k=-1,
                                every_n_epochs=1,
                            )],
                        check_val_every_n_epoch=1,)  # TODO - add default_root_dir?

    # load exisitng weights    
    if jax_param_path:
        msasaxsmodel.load_from_jax(jax_param_path)
        logging.info(f"Successfully loaded JAX parameters at {jax_param_path}...")
    
    if resume_model_weights_only:
        msasaxsmodel.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'], strict=False)
        ckpt_path= None
        msasaxsmodel.ema = ExponentialMovingAverage(
            model=msasaxsmodel.model, decay=config.ema.decay
        ) # need to initialize EMA this way at the beginning

    # fit the model
    trainer.fit(model=msasaxsmodel, train_dataloaders=train_loader, ckpt_path=ckpt_path)
    # trainer.fit(model=msasaxsmodel, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)