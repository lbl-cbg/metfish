
import time
import torch
import pytorch_lightning as pl

from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.model.model import AlphaFold

from metfish.msa_model.utils.loss import compute_saxs


class SAXSLoss(torch.nn.Module):
    """SAXS loss for MSA refinement"""
    def __init__(self, config):
        super(SAXSLoss, self).__init__()
        self.config = config
    
    def forward(self, out, batch):
        all_atom_pred_pos = out["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
        all_atom_true_pos = batch["all_atom_positions"]
        step = self.config.saxs_loss.step
        dmax = self.config.saxs_loss.dmax

        # calculate predicted and true saxs
        pred_saxs = compute_saxs(all_atom_pos=all_atom_pred_pos, all_atom_mask=all_atom_mask, step=step, dmax=dmax)
        true_saxs = compute_saxs(all_atom_pos=all_atom_true_pos, all_atom_mask=all_atom_mask, step=step, dmax=dmax)

        # get L1 loss
        l1_loss = torch.nn.L1Loss(reduction="sum")
        loss = l1_loss(pred_saxs, true_saxs)

        return loss


class MSARefinementModel(torch.nn.Module):
    def __init__(self, config, training=True):
        super(MSARefinementModel, self).__init__()
        self.config = config
        self.af_model = AlphaFold(config)
        self.training = training
        
        # freeze AF parameters to allow gradient flow through AF model
        for param in self.af_model.parameters():
            param.requires_grad = False

    def initialize_parameters(self, msa):
        self.w = torch.nn.Parameter(torch.ones_like(msa))
        self.b = torch.nn.Parameter(torch.zeros_like(msa))

    def forward(self, batch):
        device = batch['aatype'].device
        self.w = self.w.to(device)
        self.b = self.b.to(device)

        # refine msa with linear layer
        # TODO - msa cluster profile here may mean the msa features after embedding... need to modify if so
        msa_refined = self.w * batch['msa_feat'] + self.b 
        batch['msa_feat'] = msa_refined

        # run through alphafold
        outputs = self.af_model(batch)

        return outputs

# define the lightning module for training
class MSARefinementModelWrapper(pl.LightningModule):
    def __init__(self, config, training=True, num_iterations=100, lr_mul=1.0, lr_add=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = MSARefinementModel(config)
        self.training = training
        if training:
            self.loss = SAXSLoss(config.loss)
            self.cached_weights = None

            # initial learning rates from Fadini et al.
            self.lr_mul = lr_mul  
            self.lr_add = lr_add 
            self.num_iterations = num_iterations

        # activate manual optimization
        self.automatic_optimization = False
        
        self.last_log_time = time.time()

        # TODO - initialize MSA refinement parameters as part of config file
        self.model.initialize_parameters(torch.zeros(256))

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss, iter=None, train=True):
        self.log(
            f"loss_iter{iter}",
            loss,
            on_step=False, on_epoch=True, logger=True, sync_dist=True
        )
        self.log('dur', time.time() - self.last_log_time, sync_dist=True)
        self.last_log_time = time.time()

    def training_step(self, batch):
        self.model.initialize_parameters(batch['msa_feat'])
        opt = self.optimizers()

        for n in range(self.num_iterations):
            print(f'Running iteration {n} / {self.num_iterations}')
            
            # clear gradients
            opt.zero_grad()

            # forward pass
            outputs = self.model(batch)

            # calculate loss
            batch_no_recycling = tensor_tree_map(lambda t: t[..., -1], batch)  # remove recycling dimension
            loss = self.loss(outputs, batch_no_recycling) 

            # backwards pass and update weights
            self.manual_backward(loss, retain_graph=True)  # retain graph to use same computation graph multiple times
            opt.step()

            # log the loss
            self._log(loss.detach(), iter=n)

        return loss
    
    def configure_optimizers(self, eps: float = 1e-5,):
        optimizer = torch.optim.Adam([
            {'params': [self.model.w], 'lr': self.lr_mul},
            {'params': [self.model.b], 'lr': self.lr_add}
        ], eps=eps)
        
        return optimizer
        
    def load_from_jax(self, jax_path):
        import_jax_weights_(
                self.model.af_model, jax_path, version='model_3'
        )
