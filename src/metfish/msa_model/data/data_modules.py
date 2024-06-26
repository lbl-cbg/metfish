import torch

import pandas as pd

from torch.utils.data import Dataset

from metfish.msa_model.data.data_pipeline import DataPipeline
from metfish.msa_model.data.feature_pipeline import FeaturePipeline

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
      pdb_features = self.data_pipeline.process_pdb_feats(f'{self.pdb_dir}/fixed_{item.name}.pdb', is_distillation=False)
      data = {**sequence_feats, **msa_features, **saxs_features, **pdb_features}

      feats = self.feature_pipeline.process_features(data)

      feats["batch_idx"] = torch.tensor(
            [idx for _ in range(feats["aatype"].shape[-1])],
            dtype=torch.int64,
            device=feats["aatype"].device) 
              
      return feats