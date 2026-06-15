
import argparse
import torch
import os
import lightning.pytorch as pl

from pathlib import Path
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from openfold.utils.import_weights import import_jax_weights_

from metfish.msa_model.config import model_config
from metfish.msa_model.data.data_modules import MSASAXSDataset
from metfish.refinement_model.random_model import MSARandomModel
from metfish.refinement_model.model_wrapper import generate_ensemble

# gives a speedup on Ampere-class GPUs
torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default="/global/cfs/cdirs/m3513/metfish/apo_holo_data",
    help="Directory containing training pdb, saxs, and msa data",
)
parser.add_argument(
    "--output_dir", type=str, default="/pscratch/sd/l/lemonboy/metfish_output_2025/random_AF2",
    help='''Directory in which to output ensemble structures, logs, etc.''',
)
parser.add_argument(
    "--seed", type=int, default=1,
    help="Random seed"
)
parser.add_argument(
    "--use_wandb", action="store_true", default=True,
    help="Whether to log metrics to Weights & Biases"
)
parser.add_argument(
    "--jax_param_path", type=str, default="/pscratch/sd/s/smprince/projects/alphaflow/params_model_1.npz",  # these are the original AF weights,
    help="""Path to an .npz JAX parameter file with which to initialize the model"""
)
parser.add_argument(
    "--precision", type=str, default='bf16-mixed',
    help='Sets precision, lower precision improves runtime performance.',
)
parser.add_argument(
    "--job_name", type=str,
    help='''Name of job to be used for logging purposes.''',
)
parser.add_argument(
    "--num_ensemble", type=int, default=10,
    help="Number of ensemble structures to generate"
)
parser.add_argument(
    "--overwrite", default=False, action='store_true',
    help="Whether to skip generation if ensemble structures already exist"
)
def main(data_dir="/global/cfs/cdirs/m3513/metfish/apo_holo_data",
         output_dir="/pscratch/sd/l/lemonboy/metfish_output_2025/random_AF2",
         batch_size=1,
         seed=1,
         use_wandb=False,
         jax_param_path="/pscratch/sd/l/lemonboy/alphaflow_new_branch/params_model_1.npz",
         precision='bf16-mixed',
         job_name='ensemble_generation',
         num_ensemble=10,
         overwrite=False,
        ):
    
    # set up data paths and configuration
    pdb_dir = f"{data_dir}/pdbs"
    saxs_dir = f"{data_dir}/saxs_r"
    msa_dir = f"{data_dir}/msa"
    training_csv = f'{data_dir}/input_no_training_data.csv'

    pl.seed_everything(seed, workers=True) 
    config = model_config('generating', train=False, low_prec=True) 
    data_config = config.data
    data_config.common.use_templates = False
    data_config.common.max_recycling_iters = 0

    # set up datasets and dataloaders
    dataset = MSASAXSDataset(data_config, training_csv, msa_dir=msa_dir, saxs_dir=saxs_dir, pdb_dir=pdb_dir, saxs_ext='_atom_only.csv', pdb_prefix='', pdb_ext='_atom_only.pdb')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # initialize model and load existing weights
    random_model = MSARandomModel(config, training=False)
    if jax_param_path:
        import_jax_weights_(random_model.af_model, jax_param_path, version='model_3')
        print(f"Successfully loaded JAX parameters at {jax_param_path}...")

    # add logging
    Path(f"{output_dir}/lightning_logs/{job_name}").mkdir(parents=True, exist_ok=True)
    loggers = [CSVLogger(f"{output_dir}/lightning_logs/{job_name}", name=job_name)]
    if use_wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        os.environ["WANDB_MODE"] = "offline"
        loggers.append(WandbLogger(name=job_name, save_dir=f"{output_dir}/lightning_logs/{job_name}"))

    # configure fabric
    fabric = Fabric(accelerator="gpu", 
                    devices=1,
                    loggers=loggers,
                    precision=precision)

    # run ensemble generation for each sequence in dataset
    data_loader = fabric.setup_dataloaders(data_loader)
    for i, batch in enumerate(data_loader):
        seq_name = dataset.get_name(int(batch['batch_idx']))
        ensemble_output_dir = f"{output_dir}/ensemble/{job_name}/{seq_name}"
        
        # skip if ensemble already exists
        if not overwrite and os.path.exists(ensemble_output_dir) and len(os.listdir(ensemble_output_dir)) >= num_ensemble:
            print(f"Skipping {seq_name} as ensemble already exists with {len(os.listdir(ensemble_output_dir))} structures.")
            continue

        Path(ensemble_output_dir).mkdir(parents=True, exist_ok=True)

        # setup model
        model = fabric.setup(random_model)

        # generate ensemble
        print(f'Generating {num_ensemble} ensemble structures for {seq_name}')
        generate_ensemble(fabric, model, batch, 
                         num_ensemble=num_ensemble,
                         output_dir=ensemble_output_dir,
                         seq_name=seq_name)

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
