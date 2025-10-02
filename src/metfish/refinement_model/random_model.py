
import torch
from openfold.model.model import AlphaFold


from metfish.msa_model.utils.loss import compute_saxs,saxs_loss
from metfish.refinement_model.model.saxs_structure import StructureSAXS

def compute_plddt_loss(plddt_scores, target_plddt=90.0, loss_type='mae', seq_length=None):
    """
    Compute pLDDT loss to encourage high confidence predictions
    
    Args:
        plddt_scores: Predicted pLDDT scores [*, N_res]
        target_plddt: Target pLDDT value (default: 90.0 for high confidence)
        loss_type: Type of loss - 'mse', 'mae', or 'huber'
        seq_length: Actual sequence length to cutoff plddt_scores (optional)
    
    Returns:
        pLDDT loss scalar (normalized by number of residues)
    """
    # Apply sequence length cutoff if provided
    if seq_length is not None:
        # Ensure we don't exceed the available plddt_scores length
        actual_length = min(seq_length, plddt_scores.shape[-1])
        plddt_scores = plddt_scores[..., :actual_length]
    
    target = torch.full_like(plddt_scores, target_plddt)
    
    if loss_type == 'mse':
        loss = torch.nn.functional.mse_loss(plddt_scores, target, reduction='mean')
    elif loss_type == 'mae':
        loss = torch.nn.functional.l1_loss(plddt_scores, target, reduction='mean')
    elif loss_type == 'huber':
        loss = torch.nn.functional.huber_loss(plddt_scores, target, reduction='mean')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss

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
    
    def compute_loss(self, outputs, batch, plddt_weight=0.1, plddt_target=90.0):
        """Compute combined SAXS + pLDDT loss for training"""
        # Extract predicted atom positions
        pred_positions = outputs['final_atom_positions']  # [*, N, 37, 3]
        
        # Compute SAXS loss
        saxs_loss_calculated = saxs_loss(all_atom_pred_pos=pred_positions, all_atom_mask=outputs.get('final_atom_mask', None), saxs=batch['saxs'])
        
        # Compute pLDDT loss
        plddt_scores = outputs.get('plddt', None)
        if plddt_scores is not None:
            # Get sequence length from batch
            seq_length = batch.get('seq_length', None)
            if seq_length is not None:
                try:
                    # Handle various tensor formats: [[134, 134]], [134], or 134
                    seq_length = seq_length.squeeze().item()
                except (ValueError, RuntimeError):
                    # Fallback: take first element if squeeze().item() fails
                    seq_length = seq_length.flatten()[0].item()

            plddt_loss_calculated = compute_plddt_loss(plddt_scores, target_plddt=plddt_target, loss_type='mae', seq_length=seq_length)
        else:
            plddt_loss_calculated = torch.tensor(0.0, device=pred_positions.device)
        
        # Combine losses
        total_loss = saxs_loss_calculated + plddt_weight * plddt_loss_calculated
        
        return total_loss
    
    def training_step(self, batch, optimizer, plddt_weight=0.1, plddt_target=90.0):
        """Single training step"""
        # Forward pass
        outputs = self.forward(batch)
        
        # Compute loss
        loss = self.compute_loss(outputs, batch, plddt_weight=plddt_weight, plddt_target=plddt_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, optimizer, device='cuda', plddt_weight=0.1, plddt_target=90.0):
        """Train for one epoch"""
        self.structure_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Training step
            loss = self.training_step(batch, optimizer, plddt_weight=plddt_weight, plddt_target=plddt_target)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                print(f"Batch {batch_idx}, Loss: {loss:.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, dataloader, device='cuda', plddt_weight=0.1, plddt_target=90.0):
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
                loss = self.compute_loss(outputs, batch, plddt_weight=plddt_weight, plddt_target=plddt_target)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
