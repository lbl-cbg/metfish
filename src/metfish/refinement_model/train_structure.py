#!/usr/bin/env python3
"""
Runtime optimization script for StructureModel with single sequence/SAXS pair
Optimizes SingleOptimizer parameters to fit a single sequence to its SAXS curve
"""

import argparse
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Import your existing modules
from metfish.msa_model.config import model_config
from metfish.msa_model.data.data_modules import MSASAXSDataset
from metfish.refinement_model.random_model import StructureModel
from metfish.utils import output_to_protein
from openfold.np import protein
from openfold.np.relax import relax
import numpy as np


def save_structure_output(outputs, batch, output_path, seq_name, iteration=None, 
                         apply_relaxation=False, use_gpu=True):
    """Save predicted structure to PDB file with optional AMBER relaxation"""
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
    
    # Apply AMBER relaxation if requested
    if apply_relaxation:
        print(f"Applying AMBER relaxation to structure...")
        try:
            # Initialize AMBER relaxer with OpenFold defaults
            amber_relaxer = relax.AmberRelaxation(
                max_iterations=0,  # 0 means no max limit
                tolerance=2.39,    # kcal/mol energy tolerance
                stiffness=10.0,    # kcal/mol/A^2 restraint stiffness
                exclude_residues=[], # No residues excluded from restraints
                max_outer_iterations=20,  # Maximum violation-informed iterations
                use_gpu=use_gpu
            )
            
            # Run relaxation
            relaxed_pdb_str, debug_data, violations = amber_relaxer.process(
                prot=unrelaxed_protein
            )
            
            print(f"Relaxation completed:")
            print(f"  - Initial energy: {debug_data['initial_energy']:.2f} kcal/mol")
            print(f"  - Final energy: {debug_data['final_energy']:.2f} kcal/mol")
            print(f"  - RMSD from initial: {debug_data['rmsd']:.3f} Å")
            print(f"  - Attempts: {debug_data['attempts']}")
            print(f"  - Violations remaining: {violations.sum()}")
            
            # Create relaxed protein object
            relaxed_protein = protein.from_pdb_string(relaxed_pdb_str)
            final_protein = relaxed_protein
            suffix = "_relaxed"
            
        except Exception as e:
            print(f"Relaxation failed: {e}")
            print("Saving unrelaxed structure instead...")
            final_protein = unrelaxed_protein
            suffix = "_unrelaxed"
    else:
        final_protein = unrelaxed_protein
        suffix = "_unrelaxed"
    
    # Save to PDB file
    if iteration is not None:
        pdb_path = f"{output_path}/{seq_name}_iter_{iteration:04d}{suffix}.pdb"
    else:
        pdb_path = f"{output_path}/{seq_name}_optimized{suffix}.pdb"
    
    os.makedirs(os.path.dirname(pdb_path), exist_ok=True)
    
    with open(pdb_path, 'w') as f:
        f.write(protein.to_pdb(final_protein))
    
    return pdb_path


def optimize_single_sequence(model, batch, target_saxs, seq_name, 
                           num_iterations=1000, learning_rate=1e-3, 
                           output_dir=None, save_frequency=50, device='cuda',
                           apply_relaxation=False, relax_frequency=None):
    """
    Optimize SingleOptimizer parameters for a single sequence to match SAXS curve
    
    Args:
        model: StructureModel instance
        batch: Single batch containing sequence data
        target_saxs: Target SAXS curve to fit
        seq_name: Name of the sequence
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization (will be adapted based on initial loss)
        output_dir: Directory to save outputs
        save_frequency: Save structures every N iterations
        device: Device to run on
        apply_relaxation: Whether to apply AMBER relaxation to structures
        relax_frequency: Apply relaxation every N iterations (None means only final and best)
    """
    
    print(f"\nOptimizing sequence: {seq_name}")
    print(f"Target SAXS shape: {target_saxs.shape}")
    print(f"Number of iterations: {num_iterations}")
    
    # Get initial loss to adapt learning rate and scheduler parameters
    model.structure_model.eval()  # Use eval mode for initial assessment
    with torch.no_grad():
        initial_outputs = model.forward(batch)
        initial_loss = model.compute_loss(initial_outputs, batch).item()
    
    # Adaptive learning rate based on initial loss magnitude
    if initial_loss > 0.1:
        adaptive_lr = 0.1
        factor = 0.5  # More aggressive reduction for high loss
        patience = 25  # Even shorter patience for faster adaptation
    elif initial_loss > 0.01:
        adaptive_lr = 0.01
        factor = 0.7  # Moderate reduction
        patience = 35  # Medium patience
    else:
        adaptive_lr = 0.001
        factor = 0.8  # Conservative reduction for fine-tuning
        patience = 50  # Longer patience for stable convergence
    
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Adaptive learning rate: {adaptive_lr}")
    print(f"Scheduler factor: {factor}, patience: {patience}")
    
    # Setup optimizer for SingleOptimizer parameters only
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.Adam(trainable_params, lr=adaptive_lr)
    
    # Store best model state for restoration when LR is reduced
    best_model_state = None
    best_optimizer_state = None
    
    # Custom scheduler that restores best weights before reducing LR
    class RestoreBestOnPlateauScheduler:
        def __init__(self, optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6, verbose=True):
            self.optimizer = optimizer
            self.mode = mode
            self.factor = factor
            self.patience = patience
            self.min_lr = min_lr
            self.verbose = verbose
            
            self.best_metric = float('inf') if mode == 'min' else float('-inf')
            self.num_bad_epochs = 0
            self.last_epoch = 0
            self.cooldown_counter = 0
            
        def step(self, metric, model_state=None, optimizer_state=None):
            # Update best metric and reset counter if improvement
            is_better = metric < self.best_metric if self.mode == 'min' else metric > self.best_metric
            
            if is_better:
                self.best_metric = metric
                self.num_bad_epochs = 0
                # Store best states if provided
                if model_state is not None:
                    self.best_model_state = model_state
                if optimizer_state is not None:
                    self.best_optimizer_state = optimizer_state
            else:
                self.num_bad_epochs += 1
                
            # Check if we should reduce LR
            if self.num_bad_epochs >= self.patience:
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr > self.min_lr:
                    # Restore best model state before reducing LR
                    if hasattr(self, 'best_model_state') and self.best_model_state is not None:
                        if self.verbose:
                            print(f"\nRestoring best model state (metric: {self.best_metric:.6f}) before reducing LR...")
                        return {'restore_model': True, 'reduce_lr': True}
                    else:
                        if self.verbose:
                            print(f"\nReducing LR from {current_lr:.2e} to {current_lr * self.factor:.2e}")
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
                        self.num_bad_epochs = 0
                        
            return {'restore_model': False, 'reduce_lr': False}
    
    # Create custom scheduler
    scheduler = RestoreBestOnPlateauScheduler(
        optimizer, mode='min', factor=factor, patience=patience, min_lr=1e-6, verbose=True
    )
    
    model.structure_model.train()
    
    # Track optimization progress
    loss_history = []
    best_loss = float('inf')
    best_iteration = 0
    best_model_state = None
    best_optimizer_state = None
    
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
        
        # Update best loss and save best model state
        if loss_value < best_loss:
            best_loss = loss_value
            best_iteration = iteration
            
            # Store best model and optimizer states
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_optimizer_state = {k: v.clone() if torch.is_tensor(v) else v 
                                  for k, v in optimizer.state_dict().items()}
            
            # Save best structure (with relaxation if requested)
            if output_dir:
                best_structure_dir = f"{output_dir}/best"
                # Apply relaxation only for best structures or at specified frequency
                should_relax = apply_relaxation and (
                    relax_frequency is None or iteration % relax_frequency == 0
                )
                pdb_path = save_structure_output(
                    outputs, batch, best_structure_dir, seq_name, iteration,
                    apply_relaxation=should_relax, use_gpu=('cuda' in device)
                )
                # Save best model parameters
                torch.save({
                    'iteration': iteration,
                    'loss': loss_value,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{output_dir}/best_model.pth")
        
        # Learning rate scheduling with best weight restoration
        scheduler_action = scheduler.step(best_loss, best_model_state, best_optimizer_state)
        
        # Restore best weights if scheduler requests it
        if scheduler_action['restore_model'] and best_model_state is not None:
            print(f"Restoring best model state from iteration {best_iteration}")
            model.load_state_dict(best_model_state)
            optimizer.load_state_dict(best_optimizer_state)
            
            # Now reduce learning rate
            if scheduler_action['reduce_lr']:
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * factor, 1e-6)
                print(f"Reducing LR from {current_lr:.2e} to {new_lr:.2e}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                # Reset scheduler patience counter
                scheduler.num_bad_epochs = 0
        
        # Progress logging
        if iteration % max(1, num_iterations // 20) == 0 or iteration < 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"{iteration:8d} | {loss_value:8.6f} | {best_loss:8.6f} | {current_lr:.2e}")
        
        # Save intermediate structures
        if output_dir and save_frequency > 0 and iteration % save_frequency == 0:
            intermediate_dir = f"{output_dir}/intermediate"
            # Optionally apply relaxation to intermediate structures
            should_relax = apply_relaxation and (
                relax_frequency is not None and iteration % relax_frequency == 0
            )
            save_structure_output(
                outputs, batch, intermediate_dir, seq_name, iteration,
                apply_relaxation=should_relax, use_gpu=('cuda' in device)
            )
        
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
    
    # Save final structure and results (always apply relaxation if requested)
    if output_dir:
        final_dir = f"{output_dir}/final"
        final_pdb = save_structure_output(
            outputs, batch, final_dir, seq_name,
            apply_relaxation=apply_relaxation, use_gpu=('cuda' in device)
        )
        
        # Save loss history
        np.save(f"{output_dir}/loss_history.npy", np.array(loss_history))
        
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
    parser.add_argument("--pdb_ext", type=str, default=".pdb",
                       help="PDB file extension")
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
    parser.add_argument("--random_init", action="store_true",
                       help="Randomly initialize SingleOptimizer parameters")
    parser.add_argument("--apply_relaxation", action="store_true",
                       help="Apply AMBER relaxation to predicted structures")
    parser.add_argument("--relax_frequency", type=int, default=None,
                       help="Apply relaxation every N iterations (None means only final and best)")
    
    args = parser.parse_args()
    
    # Setup paths
    pdb_dir = f"{args.data_dir}/pdbs"
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
        pdb_dir=pdb_dir, 
        saxs_ext=args.saxs_ext, 
        pdb_prefix='', 
        pdb_ext=args.pdb_ext
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
    
    # Save initial structure (will be calculated again in optimize_single_sequence)
    with torch.no_grad():
        initial_outputs = model.forward(batch)
        initial_dir = f"{args.output_dir}/initial"
        save_structure_output(initial_outputs, batch, initial_dir, seq_name,
                            apply_relaxation=False)  # Don't relax initial structure
    
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
        apply_relaxation=args.apply_relaxation,
        relax_frequency=args.relax_frequency
    )
    
    # Get final loss for comparison
    with torch.no_grad():
        model.structure_model.eval()
        final_outputs = model.forward(batch)
        final_loss = model.compute_loss(final_outputs, batch).item()
    
    print(f"\nOptimization Results:")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Best loss during training: {best_loss:.6f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
