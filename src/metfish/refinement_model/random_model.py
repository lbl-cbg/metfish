
import torch
from openfold.model.model import AlphaFold


from metfish.msa_model.utils.loss import compute_saxs,saxs_loss
from metfish.refinement_model.model.saxs_structure import StructureSAXS

# --- Copied from train_structure.py to avoid circular import ---
from metfish.utils import output_to_protein
from openfold.np import protein
import os

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



class MSARandomModel(torch.nn.Module):
    def __init__(self, config, training=False):
        super(MSARandomModel, self).__init__()
        self.config = config
        self.af_model = AlphaFold(config)
        for param in self.af_model.parameters():
            param.requires_grad = False
                    
    def initialize_parameters(self, msa):
        self.w = torch.nn.Parameter(torch.randn_like(msa) * 0.1)
        self.b = torch.nn.Parameter(torch.randn_like(msa) * 0.1)

    def forward(self, batch):
        noise = torch.randn_like(batch['msa_feat']) * 0.01
        msa_feat_refined = self.w * batch['msa_feat'] + self.b + noise 
        batch.update({'msa_feat': msa_feat_refined})
        outputs = self.af_model(batch)

        return outputs
    
class StructureModel(torch.nn.Module):
    def __init__(self, config, training=False):
        super(StructureModel, self).__init__()
        self.config = config
        self.training_mode = training
        
        # Use StructureSAXS model instead of basic AlphaFold
        self.structure_model = StructureSAXS(config)
        
        # Freeze all parameters except SingleOptimizer
        for name, param in self.structure_model.named_parameters():
            if 'single_optimizer' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    def load_pretrained_weights(self, checkpoint_path, strict=False):
        """
        Load pretrained weights from PyTorch Lightning checkpoint
        
        Args:
            checkpoint_path: Path to the .ckpt file
            strict: Whether to strictly enforce that the keys match
        """
        print(f"Loading pretrained weights from {checkpoint_path}")
        
        # Load PyTorch Lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict from Lightning checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"Found state_dict with {len(state_dict)} keys")
        else:
            # Fallback if it's a regular torch checkpoint
            state_dict = checkpoint
            print(f"Using checkpoint directly with {len(state_dict)} keys")
        
        # Remove 'model.' prefix if present (common in Lightning checkpoints)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value
        
        # Load weights into the structure model
        load_result = self.structure_model.load_state_dict(
            cleaned_state_dict, strict=strict
        )
        print("Pretrained weights loaded successfully!")
        if hasattr(load_result, 'missing_keys') and hasattr(load_result, 'unexpected_keys'):
            print(f"Missing keys: {load_result.missing_keys}")
            print(f"Unexpected keys: {load_result.unexpected_keys}")
        else:
            print(f"load_state_dict result: {load_result}")

        # Re-freeze parameters except SingleOptimizer
        for name, param in self.structure_model.named_parameters():
            if 'single_optimizer' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        print(f"Trainable parameters: {len(self.get_trainable_parameters())}")
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters (SingleOptimizer)"""
        trainable_params = []
        for name, param in self.structure_model.named_parameters():
            if 'single_optimizer' in name and param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def forward(self, batch):
        """Forward pass through the structure model"""
        outputs = self.structure_model(batch)
        return outputs
    
    def compute_loss(self, outputs, batch):
        """Compute SAXS loss for training"""
        # Extract predicted atom positions
        pred_positions = outputs['final_atom_positions']  # [*, N, 37, 3]
        
        # Compute SAXS curve from predicted structure

        saxs_loss_calculated= saxs_loss(all_atom_pred_pos=pred_positions, all_atom_mask=outputs.get('final_atom_mask', None), saxs=batch['saxs'])

        return saxs_loss_calculated
    
    def training_step(self, batch, optimizer):
        """Single training step"""
        # Forward pass
        outputs = self.forward(batch)
        
        # Compute loss
        loss = self.compute_loss(outputs, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, optimizer, device='cuda'):
        """Train for one epoch"""
        self.structure_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Training step
            loss = self.training_step(batch, optimizer)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                print(f"Batch {batch_idx}, Loss: {loss:.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, dataloader, device='cuda'):
        """Validation loop"""
        self.structure_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.forward(batch)
                
                # Compute loss
                loss = self.compute_loss(outputs, batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss


def train_structure_model(model, train_loader, val_loader, num_epochs=100, 
                         learning_rate=1e-4, device='cuda', save_path=None, output_dir=None):
    """
    Complete training loop for StructureModel
    
    Args:
        model: StructureModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        save_path: Path to save the best model
    """
    
    # Setup optimizer - only optimize SingleOptimizer parameters
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Move model to device
    model = model.to(device)
    
    best_eval_loss = float('inf')
    
    print(f"Starting training with {len(trainable_params)} trainable parameters")
    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        # Training
        train_loss = model.train_epoch(train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.6f}")

        # Evaluation (save PDB for first batch, keep model in training mode)
        eval_loss = 0.0
        num_batches = 0
        for batch in val_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model.forward(batch)
            loss = model.compute_loss(outputs, batch)
            eval_loss += loss.item()
            num_batches += 1
            # Save first batch for PDB output (or you can save all if desired)
            if num_batches == 1 and output_dir is not None:
                pdb_dir_epoch = f"{output_dir}/epoch_pdb"
                seq_name_epoch = f'eval_sample_epoch{epoch+1}'
                save_structure_output(outputs, batch, pdb_dir_epoch, seq_name_epoch, iteration=epoch)
        eval_loss /= max(num_batches, 1)
        print(f"Evaluation Loss: {eval_loss:.6f}")
        scheduler.step(eval_loss)

        # Save best model and PDB
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                }, save_path)
                print(f"Saved best model with evaluation loss: {eval_loss:.6f}")
            # Save PDB for best model
            if output_dir is not None:
                pdb_dir = f"{output_dir}/best_pdb"
                seq_name = 'best_eval_sample'
                save_structure_output(outputs, batch, pdb_dir, seq_name, iteration=epoch)

        # Early stopping
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print("Learning rate too small, stopping training")
            break

    print(f"\nTraining completed. Best evaluation loss: {best_eval_loss:.6f}")
    if output_dir is not None:
        print(f"Best model PDB saved to: {output_dir}/best_pdb")
    return model


# Example usage:
if __name__ == "__main__":
    # Example of how to use the training loop with Lightning checkpoint
    
    # Initialize model
    # config = your_config_object
    # model = StructureModel(config, training=True)
    
    # Load pretrained Lightning weights
    # checkpoint_path = "epoch=15-step=21009.ckpt"
    # model.load_pretrained_weights(checkpoint_path, strict=False)
    
    # Setup data loaders
    # train_loader = your_train_dataloader
    # val_loader = your_val_dataloader
    
    # Train the model (only SingleOptimizer parameters will be trained)
    # trained_model = train_structure_model(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=100,
    #     learning_rate=1e-4,
    #     device='cuda',
    #     save_path='best_structure_model.pth'
    # )
    
    pass