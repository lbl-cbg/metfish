
import torch
from openfold.model.model import AlphaFold

from metfish.msa_model.utils.loss import compute_saxs



class MSARandomModel(torch.nn.Module):
    def __init__(self, config, training=False):
        super(MSARandomModel, self).__init__()
        self.config = config
        self.af_model = AlphaFold(config)
        for param in self.af_model.parameters():
            param.requires_grad = False
                    
    def initialize_parameters(self, msa):
        self.w = torch.nn.Parameter(torch.randn_like(msa) * 0.1)
        self.b = torch.nn.Parameter(torch.randn_like(msa) * 0.1)

    def forward(self, batch):
        noise = torch.randn_like(batch['msa_feat']) * 0.01
        msa_feat_refined = self.w * batch['msa_feat'] + self.b + noise 
        batch.update({'msa_feat': msa_feat_refined})
        outputs = self.af_model(batch)

        return outputs