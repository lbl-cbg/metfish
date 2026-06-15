#!/usr/bin/env python3
"""
Runtime optimization script for StructureModel with single sequence/SAXS pair
Optimizes SingleOptimizer parameters to fit a single sequence to its SAXS curve

Output Structure:
- best/: Contains the best structures (when loss improves) + best_model.pth
- intermediate/: Contains ALL generated structures during training (if save_all_structures=True)
                or structures saved at regular intervals (if save_interval_only=True)
- initial/: Contains the initial structure before optimization
- final/: Contains the final structure after optimization
- loss_history.npy: Array of loss values for each iteration
- optimization_summary.txt: Summary of optimization results
"""

import argparse
import os
import torch
import csv
from pathlib import Path
from torch.utils.data import DataLoader

# Import your existing modules
from metfish.msa_model.config import model_config
from metfish.msa_model.data.data_modules import MSASAXSDataset
from metfish.refinement_model.random_model import StructureModel
from metfish.utils import output_to_protein
from openfold.np import protein
import numpy as np


def save_structure_output(outputs, batch, output_path, seq_name, iteration=None):
    """Save predicted structure to PDB file"""
    # Extract relevant information for structure output
    out_to_prot_keys = ['final_atom_positions', 'final_atom_mask', 'aatype', 'seq_length', 'residue_index', 'plddt']

    output_info = {}
    for key in out_to_prot_keys:
        if key in outputs:
            output_info[key] = outputs[key].clone().detach().cpu()
        elif key in batch:
            output_info[key] = batch[key].clone().detach().cpu()
    
    # Create protein object
    unrelaxed_protein = output_to_protein(output_info)
    
    # Save to PDB file
    if iteration is not None:
        pdb_path = f"{output_path}/{seq_name}_iter_{iteration:04d}.pdb"
    else:
        pdb_path = f"{output_path}/{seq_name}_optimized.pdb"
    
    os.makedirs(os.path.dirname(pdb_path), exist_ok=True)
    
    with open(pdb_path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))
    
    return pdb_path


def optimize_single_sequence(model, batch, target_saxs, seq_name, 
                           num_iterations=1000, learning_rate=1e-3, 
                           output_dir=None, save_frequency=50, device='cuda',
                           save_all_structures=True):
    """
    Optimize SingleOptimizer parameters for a single sequence to match SAXS curve
    
    Args:
        model: StructureModel instance
        batch: Single batch containing sequence data
        target_saxs: Target SAXS curve to fit
        seq_name: Name of the sequence
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        output_dir: Directory to save outputs
        save_frequency: Save structures every N iterations (for checkpoints)
        device: Device to run on
        save_all_structures: If True, save all structures to intermediate folder
    """
    
    print(f"\nOptimizing sequence: {seq_name}")
    print(f"Target SAXS shape: {target_saxs.shape}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Learning rate: {learning_rate}")
    if save_all_structures:
        print(f"WARNING: All structures will be saved to intermediate folder - this may use significant disk space!")
        print(f"Expected storage: ~{num_iterations * 0.5:.1f} MB (approximate)")
    else:
        print(f"Structures will be saved every {save_frequency} iterations to intermediate folder")
    
    # Setup optimizer for SingleOptimizer parameters only
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
    # Optional learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-6, verbose=True
    )
    
    model.structure_model.train()
    
    # Track optimization progress
    loss_history = []
    saved_pdb_paths = []    # Track paths of saved PDB files
    saved_structure_losses = []  # Track losses when structures were saved
    best_loss = float('inf')
    best_iteration = 0
    
    print("\nStarting optimization...")
    print("Iteration | Loss      | Best Loss | LR")
    print("-" * 45)
    
    for iteration in range(num_iterations):
        # Forward pass
        optimizer.zero_grad()
        outputs = model.forward(batch)
        
        # Compute SAXS loss
        loss = model.compute_loss(outputs, batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # Update best loss
        if loss_value < best_loss:
            best_loss = loss_value
            best_iteration = iteration
            
            # Save best structure
            if output_dir:
                best_structure_dir = f"{output_dir}/best"
                pdb_path = save_structure_output(outputs, batch, best_structure_dir, 
                                               seq_name, iteration)
                # Track this saved structure
                saved_pdb_paths.append(pdb_path)
                saved_structure_losses.append(loss_value)
                # Save best model parameters
                torch.save({
                    'iteration': iteration,
                    'loss': loss_value,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{output_dir}/best_model.pth")
        
        # Learning rate scheduling
        if iteration % 20 == 0:
            scheduler.step(loss_value)
        
        # Progress logging
        if iteration % max(1, num_iterations // 20) == 0 or iteration < 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"{iteration:8d} | {loss_value:8.6f} | {best_loss:8.6f} | {current_lr:.2e}")
        
        # Save structures based on configuration
        if output_dir:
            if save_all_structures:
                # Save ALL structures in intermediate folder
                intermediate_dir = f"{output_dir}/intermediate"
                pdb_path = save_structure_output(outputs, batch, intermediate_dir, seq_name, iteration)
                # Track this saved structure
                saved_pdb_paths.append(pdb_path)
                saved_structure_losses.append(loss_value)
            elif save_frequency > 0 and iteration % save_frequency == 0:
                # Save structures at specific intervals only
                intermediate_dir = f"{output_dir}/intermediate"
                pdb_path = save_structure_output(outputs, batch, intermediate_dir, seq_name, iteration)
                # Track this saved structure
                saved_pdb_paths.append(pdb_path)
                saved_structure_losses.append(loss_value)
        # Early stopping if loss is very small
        if loss_value < 1e-8:
            print(f"\nLoss converged to {loss_value:.2e} at iteration {iteration}")
            break
        
        # Early stopping if no improvement for a long time
        if iteration - best_iteration > num_iterations // 4:
            print(f"\nNo improvement for {iteration - best_iteration} iterations. Stopping.")
            break
    
    print(f"\nOptimization completed!")
    print(f"Best loss: {best_loss:.6f} at iteration {best_iteration}")
    
    # Save final structure and results
    if output_dir:
        final_dir = f"{output_dir}/final"
        final_pdb = save_structure_output(outputs, batch, final_dir, seq_name)
        
        # Save loss history for all iterations
        np.save(f"{output_dir}/loss_history.npy", np.array(loss_history))
        
        # Save saved structure data as numpy arrays
        np.save(f"{output_dir}/saved_structure_losses.npy", np.array(saved_structure_losses))
        
        # Save CSV with saved structure data
        with open(f"{output_dir}/saved_structures.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['pdb_filename', 'loss', 'full_path'])  # header
            for pdb_path, loss in zip(saved_pdb_paths, saved_structure_losses):
                pdb_filename = os.path.basename(pdb_path)
                writer.writerow([pdb_filename, f"{loss:.8f}", pdb_path])
        
        # Save optimization summary
        with open(f"{output_dir}/optimization_summary.txt", 'w') as f:
            f.write(f"Sequence: {seq_name}\n")
            f.write(f"Total iterations: {iteration + 1}\n")
            f.write(f"Best loss: {best_loss:.6f}\n")
            f.write(f"Best iteration: {best_iteration}\n")
            f.write(f"Final loss: {loss_value:.6f}\n")
            f.write(f"Final structure: {final_pdb}\n")
        
        print(f"Results saved to: {output_dir}")
    
    return model, loss_history, best_loss


def main():
    parser = argparse.ArgumentParser(description="Optimize StructureModel for single sequence-SAXS pair")
    
    # Data arguments (matching predict.py style)
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing input data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save optimization outputs and structures")
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to pretrained Lightning checkpoint")
    parser.add_argument("--test_csv_name", type=str, default="input.csv",
                       help="Name of the CSV file with dataset info (should contain single sequence)")
    parser.add_argument("--saxs_ext", type=str, default="_atom_only.csv",
                       help="SAXS file extension")
    
    # Optimization arguments
    parser.add_argument("--num_iterations", type=int, default=1000,
                       help="Number of optimization iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate for optimization")
    parser.add_argument("--sequence_index", type=int, default=0,
                       help="Index of sequence to optimize (if multiple in CSV)")
    parser.add_argument("--save_frequency", type=int, default=50,
                       help="Save intermediate structures every N iterations")
    parser.add_argument("--save_all_structures", action="store_true", default=True,
                       help="Save all structures to intermediate folder (default: True)")
    parser.add_argument("--save_interval_only", action="store_true",
                       help="Only save structures at intervals (overrides save_all_structures)")
    parser.add_argument("--random_init", action="store_true",
                       help="Randomly initialize SingleOptimizer parameters")
    
    args = parser.parse_args()
    
    # Setup paths
    saxs_dir = f"{args.data_dir}/saxs_r"
    msa_dir = f"{args.data_dir}/msa"
    training_csv = f"{args.test_csv_name}"
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {args.data_dir}")
    print(f"Input CSV: {training_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint path: {args.ckpt_path}")
    
    # Setup model configuration
    config = model_config('generating', train=True, low_prec=True) 
    data_config = config.data
    data_config.common.use_templates = False
    data_config.common.max_recycling_iters = 1
    
    # Create dataset
    dataset = MSASAXSDataset(
        data_config, 
        training_csv, 
        msa_dir=msa_dir, 
        saxs_dir=saxs_dir, 
        saxs_ext=args.saxs_ext, 
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get single sequence data
    if args.sequence_index >= len(dataset):
        raise ValueError(f"Sequence index {args.sequence_index} out of range (dataset size: {len(dataset)})")
    
    # Create data loader for single sequence
    single_dataset = torch.utils.data.Subset(dataset, [args.sequence_index])
    data_loader = DataLoader(single_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Get the single batch
    batch = next(iter(data_loader))
    seq_name = dataset.get_name(args.sequence_index) if hasattr(dataset, 'get_name') else f"seq_{args.sequence_index}"
    target_saxs = batch['saxs']
    
    print(f"Optimizing sequence: {seq_name}")
    
    # Initialize model
    model = StructureModel(config, training=True)  # Use eval mode for base model
    
    # Load pretrained weights
    model.load_pretrained_weights(args.ckpt_path, strict=False)
    
    # Randomly initialize SingleOptimizer if requested
    if args.random_init:
        print("Randomly initializing SingleOptimizer parameters...")
        with torch.no_grad():
            for name, param in model.structure_model.named_parameters():
                if 'single_optimizer' in name:
                    if 'weight' in name and param.dim() >= 2:
                        # Xavier initialization for weights with 2+ dimensions
                        torch.nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        # Zero initialization for bias terms
                        torch.nn.init.zeros_(param)
                    else:
                        # Default normal initialization for other parameters
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Move batch to device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    
    print(f"Running optimization on {device}")
    
    # Save initial structure
    with torch.no_grad():
        initial_outputs = model.forward(batch)
        initial_loss = model.compute_loss(initial_outputs, batch).item()
        initial_dir = f"{args.output_dir}/initial"
        save_structure_output(initial_outputs, batch, initial_dir, seq_name)
        print(f"Initial loss: {initial_loss:.6f}")
    
    # Determine structure saving behavior
    save_all_structures = args.save_all_structures and not args.save_interval_only
    
    # Run optimization
    optimized_model, loss_history, best_loss = optimize_single_sequence(
        model=model,
        batch=batch,
        target_saxs=target_saxs,
        seq_name=seq_name,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        save_frequency=args.save_frequency,
        device=device,
        save_all_structures=save_all_structures
    )
    
    print(f"\nOptimization Results:")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {best_loss:.6f}")
    print(f"Improvement: {((initial_loss - best_loss) / initial_loss * 100):.2f}%")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
