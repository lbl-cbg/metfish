
import torch
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
        if training:
            self.loss = SAXSLoss(config.loss)
        
        # freeze AF parameters to allow gradient flow through AF model
        for param in self.af_model.parameters():
            param.requires_grad = False

    def initialize_parameters(self, msa):
        self.w = torch.nn.Parameter(torch.ones_like(msa))
        self.b = torch.nn.Parameter(torch.zeros_like(msa))

    def forward(self, batch):
        # refine msa with linear layer
        msa_feat_refined = self.w * batch['msa_feat'] + self.b
        batch.update({'msa_feat': msa_feat_refined})

        # run through alphafold
        outputs = self.af_model(batch)

        return outputs