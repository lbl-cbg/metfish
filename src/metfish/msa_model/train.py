
import argparse
import os
import torch
import pytorch_lightning as pl

from pytorch_lightning.profilers import PyTorchProfiler #, SimpleProfiler, AdvancedProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from openfold.utils.exponential_moving_average import ExponentialMovingAverage

from metfish.msa_model.config import model_config
from metfish.msa_model.data.data_modules import MSASAXSDataset
from metfish.msa_model.model.msa_saxs import MSASAXSModel

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
    "--resume_model_weights_only", default=False, action='store_true',
    help="Whether to load just model weights as opposed to training state"
)
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true',
    help="Whether to run a fast dev run of a single batch for testing purposes"
)
parser.add_argument(
    "--validate_only", default=False, action='store_true',
    help='''Runs only validation step, useful to collect metrics from new model / checkpoint''',
)
parser.add_argument(
    "--jax_param_path", type=str, default="/pscratch/sd/s/smprince/projects/alphaflow/params_model_1.npz",  # these are the original AF weights,
    help="""Path to an .npz JAX parameter file with which to initialize the model"""
)
parser.add_argument(
    "--deterministic", default=False, action='store_true',
    help="Whether to use deterministic algorithm for msa fraction replacement"
)
parser.add_argument(
    "--profile", default=False, action='store_true',
    help="Whether to use a profiler or not"
)
parser.add_argument(
    "--checkpoint_every_n_epochs", type=int, default=1,
    help="""Number of epochs after which to checkpoint the model"""
)
parser.add_argument(
    "--checkpoint_every_n_steps", type=int, default=500,
    help="""Number of training steps after which to checkpoint the model"""
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
    "--max_epochs", type=int, default=100,
)
parser.add_argument(
    "--log_every_n_steps", type=int, default=25,
)
parser.add_argument(
    "--num_sanity_val_steps", type=int, default=0,
)
parser.add_argument(
    "--reload_dataloaders_every_n_epochs", type=int, default=1,
)
parser.add_argument(
    "--unfreeze_af_weights", default=False, action='store_true',
)

def main(data_dir="/global/cfs/cdirs/m3513/metfish/PDB70_verB_fixed_data/result",
         output_dir="/pscratch/sd/s/smprince/projects/metfish/model_outputs",
         ckpt_path=None,
         gpus_per_node=1,
         num_nodes=1,
         batch_size=2,
         seed=1,
         use_wandb=True,
         resume_model_weights_only=False,
         fast_dev_run=False,
         validate_only=False,
         jax_param_path="/pscratch/sd/s/smprince/projects/alphaflow/params_model_1.npz",
         deterministic=False,
         profile=False,
         checkpoint_every_n_epochs=1,
         checkpoint_every_n_steps=500,
         resume_from_ckpt=False,
         precision='bf16-mixed',
         max_epochs=100,
         log_every_n_steps=25,
         num_sanity_val_steps=0,
         reload_dataloaders_every_n_epochs=1,
         unfreeze_af_weights=False,
        ):
    
    # set up data paths and configuration
    pdb_dir = f"{data_dir}/pdb"
    saxs_dir = f"{data_dir}/saxs_r"
    msa_dir = f"{data_dir}/msa"
    training_csv = f'{msa_dir}/input_training.csv'
    val_csv = f'{msa_dir}/input_validation.csv'

    pl.seed_everything(seed, workers=True) 
    strategy = "ddp" if (gpus_per_node > 1) or num_nodes > 1 else "auto"
    config = model_config('initial_training', train=True, low_prec=True) 
    if deterministic:
        config.data.eval.masked_msa_replace_fraction = 0.0
        config.model.global_config.deterministic = True
    data_config = config.data
    data_config.common.use_templates = False
    data_config.common.max_recycling_iters = 0

    # set up training and test datasets and dataloaders
    train_dataset = MSASAXSDataset(data_config, training_csv, msa_dir=msa_dir, saxs_dir=saxs_dir, pdb_dir=pdb_dir)
    val_dataset = MSASAXSDataset(data_config, val_csv, msa_dir=msa_dir, saxs_dir=saxs_dir, pdb_dir=pdb_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    # initialize model
    msasaxsmodel = MSASAXSModel(config, unfreeze_af_weights=unfreeze_af_weights)

    # add logging
    loggers = [CSVLogger(f"{output_dir}/lightning_logs", name="msasaxs")]
    if use_wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        os.environ["WANDB_MODE"] = "offline"
        # wandb.init()
        loggers.append(WandbLogger(name="msasaxs", save_dir=f"{output_dir}/lightning_logs"))

    # add callbacks
    callbacks = [ModelCheckpoint(dirpath=f"{output_dir}/checkpoints", 
                                save_top_k=-1,
                                every_n_epochs=checkpoint_every_n_epochs,),
                ModelCheckpoint(dirpath=f"{output_dir}/checkpoints",
                                save_top_k=-1,
                                every_n_train_steps=checkpoint_every_n_steps,),               ]

    # add profiler
    if profile:
        profiler = PyTorchProfiler(dirpath=f"{output_dir}/lightning_logs/pytorch_profiler", filename='profile_trace.txt', emit_nvtx=True)
        # profiler = AdvancedProfiler(dirpath=f"{output_dir}/lightning_logs/advanced_profiler", filename='profile_trace')
        # profiler = SimpleProfiler(dirpath=f"{output_dir}/lightning_logs/simple_profiler", filename='profile_trace')
    else:
        profiler = None

    # initialize trainer
    trainer = pl.Trainer(accelerator="gpu", 
                         strategy=strategy,
                            max_epochs=max_epochs, 
                            gradient_clip_val=1.,
                            limit_train_batches=1.0, 
                            limit_val_batches=1.0,
                            callbacks=callbacks,
                            check_val_every_n_epoch=1,
                            logger=loggers,
                            log_every_n_steps=log_every_n_steps,
                            default_root_dir=output_dir,
                            devices=gpus_per_node,
                            num_nodes=num_nodes,
                            fast_dev_run=fast_dev_run, 
                            precision=precision,
                            profiler=profiler,
                            )

    # load existing weights
    if jax_param_path and not resume_from_ckpt:
        msasaxsmodel.load_from_jax(jax_param_path)
        print(f"Successfully loaded JAX parameters at {jax_param_path}...")
        msasaxsmodel.ema = ExponentialMovingAverage(
            model=msasaxsmodel.model, decay=config.ema.decay
        ) # need to initialize EMA this way at the beginning

    if resume_model_weights_only:
        msasaxsmodel.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        ckpt_path = None
        msasaxsmodel.ema = ExponentialMovingAverage(
            model=msasaxsmodel.model, decay=config.ema.decay
        ) # need to initialize EMA this way at the beginning
        print("Successfully loaded model weights...")
    
    ckpt_path = None if not resume_from_ckpt else ckpt_path

    # fit the model
    if validate_only:
        trainer.validate(model=msasaxsmodel, dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model=msasaxsmodel, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
    # main(fast_dev_run=True, resume_from_ckpt=True, ckpt_path="/pscratch/sd/s/smprince/projects/metfish/model_outputs/checkpoints/epoch=3-step=2092.ckpt")
