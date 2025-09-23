"""
New training script for StructureModel using internal training functions from random_model.py
Trains only the SingleOptimizer parameters on a dataset.
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from metfish.msa_model.config import model_config
from metfish.msa_model.data.data_modules import MSASAXSDataset

from metfish.refinement_model.random_model import StructureModel, train_structure_model


def main():
    parser = argparse.ArgumentParser(description="Train StructureModel (SingleOptimizer only) on a dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing input data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs and checkpoints")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pretrained Lightning checkpoint")
    parser.add_argument("--train_csv", type=str, default="train.csv", help="CSV file with training set")
    parser.add_argument("--val_csv", type=str, default="val.csv", help="CSV file with validation set")
    parser.add_argument("--pdb_ext", type=str, default=".pdb", help="PDB file extension")
    parser.add_argument("--saxs_ext", type=str, default="_atom_only.csv", help="SAXS file extension")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save best model checkpoint")
    parser.add_argument("--random_init", action="store_true", help="Randomly initialize SingleOptimizer parameters")
    args = parser.parse_args()

    # Setup paths
    pdb_dir = f"{args.data_dir}/pdbs"
    saxs_dir = f"{args.data_dir}/saxs_r"
    msa_dir = f"{args.data_dir}/msa"
    train_csv = f"{args.data_dir}/{args.train_csv}"
    val_csv = f"{args.data_dir}/{args.val_csv}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Model config
    config = model_config('generating', train=True, low_prec=True)
    data_config = config.data
    data_config.common.use_templates = False
    data_config.common.max_recycling_iters = 1

    # Datasets
    train_dataset = MSASAXSDataset(
        data_config, train_csv, msa_dir=msa_dir, saxs_dir=saxs_dir, pdb_dir=pdb_dir,
        saxs_ext=args.saxs_ext, pdb_prefix='', pdb_ext=args.pdb_ext
    )
    val_dataset = MSASAXSDataset(
        data_config, val_csv, msa_dir=msa_dir, saxs_dir=saxs_dir, pdb_dir=pdb_dir,
        saxs_ext=args.saxs_ext, pdb_prefix='', pdb_ext=args.pdb_ext
    )
    print(f"Train set size: {len(train_dataset)} | Val set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = StructureModel(config, training=True)
    model.load_pretrained_weights(args.ckpt_path, strict=False)

    # Randomly initialize SingleOptimizer if requested
    if args.random_init:
        print("Randomly initializing SingleOptimizer parameters...")
        with torch.no_grad():
            for name, param in model.structure_model.named_parameters():
                if 'single_optimizer' in name:
                    if 'weight' in name and param.dim() >= 2:
                        torch.nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param)
                    else:
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")


    # Train using internal function (PDB saving will be handled in random_model.py)
    train_structure_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_path=args.save_path or f"{args.output_dir}/best_structure_model.pth",
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
