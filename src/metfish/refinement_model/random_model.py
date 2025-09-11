
import torch
from openfold.model.model import AlphaFold

from metfish.msa_model.utils.loss import compute_saxs
from metfish.refinement_model.model.saxs_structure import StructureSAXS



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
        saxs_loss = compute_saxs(
            positions=pred_positions,
            target_saxs=batch['saxs'],
            atom_mask=outputs.get('final_atom_mask', None),
            # Add other necessary parameters based on your compute_saxs function
        )
        
        return saxs_loss
    
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
                         learning_rate=1e-4, device='cuda', save_path=None):
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
    
    best_val_loss = float('inf')
    
    print(f"Starting training with {len(trainable_params)} trainable parameters")
    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training
        train_loss = model.train_epoch(train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.6f}")
        
        # Validation
        val_loss = model.validate(val_loader, device)
        print(f"Validation Loss: {val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"Saved best model with validation loss: {val_loss:.6f}")
        
        # Early stopping
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print("Learning rate too small, stopping training")
            break
    
    print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
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