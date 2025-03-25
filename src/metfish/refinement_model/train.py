
import argparse
import torch
import os
import lightning.pytorch as pl

from pathlib import Path
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from openfold.utils.import_weights import import_jax_weights_

from metfish.msa_model.config import model_config
from metfish.msa_model.data.data_modules import MSASAXSDataset
from metfish.refinement_model.refinement_model import MSARefinementModel
from metfish.refinement_model.model_wrapper import train

# gives a speedup on Ampere-class GPUs
torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_dir", type=str,
    help="Directory containing training pdb, saxs, and msa data",
)
parser.add_argument(
    "output_dir", type=str,
    help='''Directory in which to output checkpoints, logs, etc. Ignored
            if not on rank 0''',
)
parser.add_argument(
    "--ckpt_path", type=str,
    help='''Path to a model checkpoint from which to resume training.''',
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
    "--resume_from_ckpt", default=False, action='store_true',
    help="Whether to use a model checkpoint from which to restore training state"
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
    "--save_intermediate_pdb", default=False, action='store_true',
    help="Whether to save intermediate pdb files for every step of the optimization"
)
parser.add_argument(
    "--overwrite", default=False, action='store_true',
    help="Whether to skip optimization if model checkpoints already exist"
)
def main(data_dir,
         output_dir,
         ckpt_path=None,
         batch_size=1,
         seed=1,
         use_wandb=True,
         jax_param_path="/pscratch/sd/s/smprince/projects/alphaflow/params_model_1.npz",
         resume_from_ckpt=False,
         precision='bf16-mixed',
         job_name='optimization',
         save_intermediate_pdb=False,
         overwrite=False,
        ):
    
    # set up data paths and configuration
    pdb_dir = f"{data_dir}/pdbs"
    saxs_dir = f"{data_dir}/saxs_r"
    msa_dir = f"{data_dir}/msa"
    training_csv = f'{data_dir}/input_no_training_data.csv'

    pl.seed_everything(seed, workers=True) 
    config = model_config('initial_training', train=True, low_prec=True) 
    data_config = config.data
    data_config.common.use_templates = False
    data_config.common.max_recycling_iters = 0

    # set up training and test datasets and dataloaders
    train_dataset = MSASAXSDataset(data_config, training_csv, msa_dir=msa_dir, saxs_dir=saxs_dir, pdb_dir=pdb_dir, saxs_ext='_atom_only.csv', pdb_prefix='', pdb_ext='_atom_only.pdb')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # initialize model and load existing weights if needed
    refinement_model = MSARefinementModel(config)
    if jax_param_path and not resume_from_ckpt:
        import_jax_weights_(refinement_model.af_model, jax_param_path, version='model_3')
        print(f"Successfully loaded JAX parameters at {jax_param_path}...")

    # add logging
    Path(f"{output_dir}/lightning_logs/{job_name}").mkdir(parents=True, exist_ok=True)
    loggers = [CSVLogger(f"{output_dir}/lightning_logs/{job_name}", name=job_name)]
    if use_wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        os.environ["WANDB_MODE"] = "offline"
        loggers.append(WandbLogger(name=job_name, save_dir=f"{output_dir}/lightning_logs/{job_name}"))

    # add checkpointing
    Path(f"{output_dir}/checkpoints/{job_name}").mkdir(parents=True, exist_ok=True)
    callbacks = [ModelCheckpoint(dirpath=f"{output_dir}/checkpoints/{job_name}")]
    
    # configure fabric trainer
    fabric = Fabric(accelerator="gpu", 
                    devices=1,
                    loggers=loggers,
                    callbacks=callbacks,
                    precision=precision)

    # run training for each value in dataset
    data_loader = fabric.setup_dataloaders(train_loader)
    for i, batch in enumerate(data_loader):
        # skip if file already exists
        seq_name = train_dataset.get_name(int(batch['batch_idx']))
        ckpt_path = f"{output_dir}/checkpoints/{job_name}/model_{seq_name}.ckpt"
        if not overwrite and os.path.exists(ckpt_path.replace('.ckpt', '_phase2.ckpt')):
            print(f"Skipping {seq_name} as model checkpoint already exists.")
            continue

        intermediate_output_path = None
        if save_intermediate_pdb:
            intermediate_output_path = f"{output_dir}/intermediate_files/{job_name}/model_{seq_name}"            
            Path(intermediate_output_path).mkdir(parents=True, exist_ok=True)

        # initialize parameters and optimizer for each sequence
        refinement_model.initialize_parameters(batch['msa_feat'])
        optimizers_phase1 = torch.optim.Adam([
                        {'params': [refinement_model.w], 'lr': 1.0},
                        {'params': [refinement_model.b], 'lr': 0.05}
                       ], eps=1e-5)
        optimizers_phase2 = torch.optim.Adam([
                        {'params': [refinement_model.w], 'lr': 1e-3},
                        {'params': [refinement_model.b], 'lr': 1e-3}
                    ], eps=1e-5)

        model, optimizer1, optimizer2 = fabric.setup(refinement_model, optimizers_phase1, optimizers_phase2)

        if resume_from_ckpt:
            state = {"model": model, "optimizer1": optimizer1, "optimizer2": optimizer2, "iter": 0}
            fabric.load(ckpt_path, state)

        # run training
        print(f'Running optimization for {seq_name}')
        train(fabric, model, optimizer1, optimizer2, batch, 
              ckpt_path=ckpt_path, 
              early_stopping=False, 
              intermediate_output_path=intermediate_output_path)

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
