import torch
import numpy as np
import pickle
import random
import os
 
from openfold.np import protein, residue_constants
from openfold.np.protein import Protein
from openfold.data.data_modules import OpenFoldBatchCollator

from metfish.msa_model.config import model_config
from metfish.msa_model.data.data_modules import MSASAXSDataset
from metfish.msa_model.model.msa_saxs import MSASAXSModel
from metfish.msa_model.model.alphafold_wrapper import AlphaFoldModel
from metfish.msa_model.utils.tensor_utils import tensor_tree_map


def inference(data_dir="/global/cfs/cdirs/m3513/metfish/PDB70_verB_fixed_data/result",
         output_dir="/pscratch/sd/s/smprince/projects/metfish/model_outputs",
         ckpt_path=None,
         jax_param_path="/pscratch/sd/s/smprince/projects/alphaflow/params_model_1.npz",
         deterministic=False,
         original_weights=False,
         model_name = 'MSASAXS',
         test_csv_name = 'input_no_training_data.csv',
         pdb_ext='_atom_only.pdb',
         saxs_ext='_atom_only.csv',
         tags=None,
         random_seed=None,
         overwrite=False
        ):
    
    # set up data paths and configuration
    print('Setting up data paths and configuration...')
    saxs_dir = f"{data_dir}/saxs_r"
    msa_dir = f"{data_dir}/msa"
    pdb_dir = f"{data_dir}/pdbs"
    test_csv = f'{data_dir}/{test_csv_name}'
    tags = f'_{tags}' if tags is not None else ''

    config = model_config('initial_training', train=False, low_prec=True) 
    if random_seed is None:
        random_seed = random.randrange(2 ** 32)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    if deterministic:
        config.data.eval.masked_msa_replace_fraction = 0.0
        config.data.predict.masked_msa_replace_fraction = 0.0
    data_config = config.data
    data_config.common.use_templates = False
    data_config.common.max_recycling_iters = 5

    # set up training and test datasets and dataloaders
    dataset = MSASAXSDataset(data_config,
                             test_csv,
                             mode='predict',
                             msa_dir=msa_dir,
                             pdb_dir=pdb_dir,
                             pdb_ext=pdb_ext,
                             pdb_prefix='',
                             saxs_dir=saxs_dir,
                             saxs_ext=saxs_ext)
    
    # initialize model
    print('Initializing model...')
    if model_name == 'AFSAXS':
        model = MSASAXSModel(config, training=False)
    elif model_name == 'AlphaFold':
        model = AlphaFoldModel(config)  # wrapper for OpenFold AF

    # load weight (jax params if original AF otherwise from checkpoint)
    print('Loading weights...')
    if original_weights:
        model.load_from_jax(jax_param_path)
    else:
        model = model.load_from_checkpoint(ckpt_path)
        model.load_ema_weights()
    model = model.cuda()
    model.eval()
    
    # run inference
    with torch.no_grad():
        for i, item in enumerate(dataset):
            if not overwrite and os.path.exists(f'{output_dir}/{dataset.get_name(i)}_{model_name}{tags}_unrelaxed.pdb'):
                print(f"Skipping {dataset.get_name(i)} as output already exists.")
                continue

            # prepare input features
            collate_fn = OpenFoldBatchCollator()
            batch = collate_fn([item])
            batch = tensor_tree_map(lambda x: x.cuda(), batch)  

            # run model
            print(f"Running inference on {dataset.get_name(i)}...")
            out = model(batch)

            # drop recycling dimension
            batch = tensor_tree_map(lambda t: t[0, ..., -1].cpu().numpy(), batch)

            # save unrelaxed protein as pdb file
            unrelaxed_protein = output_to_protein({**out, **batch})
            with open(f'{output_dir}/{dataset.get_name(i)}_{model_name}{tags}_unrelaxed.pdb', 'w') as f:
                f.write(protein.to_pdb(unrelaxed_protein))

            # save output dictionary
            with open(f'{output_dir}/{dataset.get_name(i)}_{model_name}{tags}_output.pkl', 'wb') as f:
                pickle.dump(out, f)

    return None

def output_to_protein(output):
    """Returns the pbd (file) string from the model given the model output."""
    output = tensor_tree_map(lambda x: x.cpu().numpy(), output)
    final_atom_positions = output['final_atom_positions']
    final_atom_mask = output["atom37_atom_exists"]
    pred = Protein(
        aatype=output["aatype"],
        atom_positions=final_atom_positions[0],
        atom_mask=final_atom_mask,
        residue_index=output["residue_index"] + 1,
        b_factors=np.repeat(output["plddt"][...,None], residue_constants.atom_type_num, axis=-1)[0],
        chain_index=output["chain_index"] if "chain_index" in output else None,
    )
    
    return pred