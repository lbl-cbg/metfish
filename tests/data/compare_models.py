
import re

import pandas as pd
import numpy as np
import itertools

from scipy.special import rel_entr
from pathlib import Path
from metfish.msa_model.predict import inference

from metfish.utils import get_rmsd, get_lddt, save_aligned_pdb, get_Pr, get_per_residue_rmsd


project_dir = Path("/pscratch/sd/l/lemonboy/NMR_model_testing")
ckpt_dir = project_dir / "checkpoints"
apo_holo_dir = "/pscratch/sd/l/lemonboy/alphaflow_test_data/apo_holo_data_steph"
#nmr_dir = apo_holo_dir"  # nmr data used for training
nmr_csv = "input_with_msa.csv"
output_dir = project_dir / "models_2025_02_03"

# get model inference dictionary
'''
model_dict = dict(
                  AFSAXS_NMRtrain=dict(model_name='AFSAXS_NMR_L1', 
                                       data_dir=apo_holo_dir,
                                       output_dir=output_dir / "apo_holo_data",  
                                       ckpt_path=f'{ckpt_dir}/epoch=10-step=14000.ckpt', 
                                       training_csv=nmr_dir / "input_training.csv",
                                       validation_csv=nmr_dir / "input_validation.csv",
                                       pdb_ext='_atom_only.pdb',
                                       tags='NMRtrain'),
)
'''

model_conformer_dict = dict(
                  AFSAXS_NMRtrain_NMReval=dict(model_name='AFSAXS',
                                               data_dir=apo_holo_dir,
                                               output_dir=output_dir,
                                               ckpt_path=f'{ckpt_dir}/epoch=9-step=13000.ckpt',
                                               tags='NMRtrain_NMReval',
                                               test_csv_name=nmr_csv))

def run_inference(model_dict):
    for model_name, model_kwargs in model_dict.items():
        inference(**model_kwargs)

def run_inference_conformer(model_conformer_dict):
    for model_name, model_kwargs in model_conformer_dict.items():
        inference(**model_kwargs)

if __name__ == '__main__':
    run_inference_conformer(model_conformer_dict)