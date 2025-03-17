
import argparse
import torch
import os
import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from metfish.msa_model.config import model_config
from metfish.msa_model.data.data_modules import MSASAXSDataset
from metfish.refinement_model.refinement_model import MSARefinementModelWrapper

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
    "--gpus_per_node", type=int, default=1, help='Number of gpus per node (will use all 4 per node on perlmutter).'
)
parser.add_argument(
    "--num_nodes", type=int, default=1, help='Number of nodes to use for training.'
)
parser.add_argument(
    "--batch_size", type=int, default=2, help='Batch size for each training step'
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
    "--fast_dev_run", default=False, action='store_true',
    help="Whether to run a fast dev run of a single batch for testing purposes"
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
    "--max_epochs", type=int, default=100,
)
parser.add_argument(
    "--log_every_n_steps", type=int, default=25,
)
parser.add_argument(
    "--job_name", type=str,
    help='''Name of job to be used for logging purposes.''',
)

def main(data_dir="/global/cfs/cdirs/m3513/metfish/PDB70_verB_fixed_data/result",
         output_dir="/pscratch/sd/s/smprince/projects/metfish/model_outputs",
         ckpt_path=None,
         gpus_per_node=1,
         num_nodes=1,
         batch_size=1,
         seed=1,
         use_wandb=True,
         fast_dev_run=False,
         jax_param_path="/pscratch/sd/s/smprince/projects/alphaflow/params_model_1.npz",
         resume_from_ckpt=False,
         precision='bf16-mixed',
         max_epochs=100,
         log_every_n_steps=1,
         job_name='default',
        ):
    
    # set up data paths and configuration
    pdb_dir = f"{data_dir}/pdb"
    saxs_dir = f"{data_dir}/saxs_r"
    msa_dir = f"{data_dir}/msa"
    csv_dir = f"{data_dir}/scripts"
    training_csv = f'{csv_dir}/input_training.csv'  # NOTE - this was msa_dir for training v_1

    pl.seed_everything(seed, workers=True) 
    strategy = "ddp" if (gpus_per_node > 1) or num_nodes > 1 else "auto"
    config = model_config('initial_training', train=True, low_prec=True) 
    data_config = config.data
    data_config.common.use_templates = False
    data_config.common.max_recycling_iters = 0

    # set up training and test datasets and dataloaders
    train_dataset = MSASAXSDataset(data_config, training_csv, msa_dir=msa_dir, saxs_dir=saxs_dir, pdb_dir=pdb_dir, saxs_ext='.pr.csv', pdb_prefix='')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # initialize model
    refinement_model = MSARefinementModelWrapper(config)

    # add logging
    Path(f"{output_dir}/lightning_logs/{job_name}").mkdir(parents=True, exist_ok=True)
    loggers = [CSVLogger(f"{output_dir}/lightning_logs/{job_name}", name=job_name)]
    if use_wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        os.environ["WANDB_MODE"] = "offline"
        loggers.append(WandbLogger(name=job_name, save_dir=f"{output_dir}/lightning_logs/{job_name}"))

    # initialize trainer
    trainer = pl.Trainer(accelerator="gpu", 
                         strategy=strategy,
                         max_epochs=max_epochs, 
                         logger=loggers,
                         log_every_n_steps=log_every_n_steps,
                         default_root_dir=output_dir,
                         devices=gpus_per_node,
                         num_nodes=num_nodes,
                         fast_dev_run=fast_dev_run, 
                         precision=precision,
                        )

    # load existing weights
    if jax_param_path and not resume_from_ckpt:
        refinement_model.load_from_jax(jax_param_path)
        print(f"Successfully loaded JAX parameters at {jax_param_path}...")

    # fit the model
    trainer.fit(model=refinement_model, train_dataloaders=train_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
