
import time
import torch
import pytorch_lightning as pl

from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.np import residue_constants
from openfold.utils.validation_metrics import (gdt_ts, gdt_ha,)
from openfold.utils.loss import lddt_ca
from openfold.utils.superimposition import superimpose
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.tensor_utils import tensor_tree_map

from metfish.msa_model.model.alphafold_saxs import AlphaFoldSAXS
from metfish.msa_model.utils.loss import AlphaFoldLossWithSAXS
from metfish.msa_model.utils.validation_metrics import drmsd

# define the lightning module for training
class MSASAXSModel(pl.LightningModule):
    def __init__(self, config, unfreeze_af_weights=False, training=True):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = AlphaFoldSAXS(config)
        self.training = training
        if training:
            self.loss = AlphaFoldLossWithSAXS(config.loss)
            self.ema = ExponentialMovingAverage(
                    model=self.model, decay=config.ema.decay
                )
            self.cached_weights = None
        self.last_lr_step = -1

        self.last_log_time = time.time()

        # freeze alphafold model
        if not unfreeze_af_weights:
            for name, param in self.model.named_parameters():
                if "saxs_msa_attention" not in name and "saxs_pair_attention" not in name:
                    param.requires_grad = False

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True, sync_dist=True
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True, sync_dist=True
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
                on_step=False, on_epoch=True, logger=True, sync_dist=True
            )
        
        self.log('dur', time.time() - self.last_log_time, sync_dist=True)
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
            def clone_param(t):
              return t.detach().clone()
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
            ema["params"] = {k:v for k,v in ema["params"].items() if  "template" not in k}
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step
        
    def load_from_jax(self, jax_path):
        import_jax_weights_(
                self.model, jax_path, version='model_3'
        )

    def load_ema_weights(self):
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling
        # load_state_dict().
        self.cached_weights = tensor_tree_map(lambda t: t.detach().clone(), self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])
