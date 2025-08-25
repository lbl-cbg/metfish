
import torch
from tqdm import tqdm
from pathlib import Path
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import protein

from metfish.utils import output_to_protein


def generate_ensemble(fabric, model, batch, 
                     num_ensemble=10,
                     output_dir=None,
                     seq_name=None):
    """
    Generate ensemble of structures using random MSA sampling
    """
    model.eval()
    
    with torch.no_grad():
        for ensemble_idx in tqdm(range(num_ensemble), desc=f"Generating ensemble for {seq_name}"):
            # create new copy of batch for each ensemble member
            iteration_batch = {k: v.clone() for k, v in batch.items()}
            
            # initialize random parameters for this ensemble member
            model.initialize_parameters(iteration_batch['msa_feat'])
            
            # forward pass with random sampling
            outputs = model(iteration_batch)
            
            # prepare output for protein generation
            batch_no_recycling = tensor_tree_map(lambda t: t[0, ..., -1], batch)
            out_to_prot_keys = ['final_atom_positions', 'plddt', 'atom37_atom_exists', 'aatype', 'residue_index', 'chain_index']
            output_info = {k: v.clone().detach() for k, v in outputs.items() if k in out_to_prot_keys}
            
            # add batch info if needed
            for k in out_to_prot_keys:
                if k in batch_no_recycling and k not in output_info:
                    output_info[k] = batch_no_recycling[k].clone().detach()
            
            unrelaxed_protein = output_to_protein(output_info)
            
            # save ensemble member
            if output_dir:
                pdb_path = f'{output_dir}/{seq_name}_ensemble_{ensemble_idx:03d}.pdb'
                with open(pdb_path, 'w') as f:
                    f.write(protein.to_pdb(unrelaxed_protein))
            
            # log progress
            fabric.log(f"ensemble_generation/{seq_name}", ensemble_idx + 1)
    
    print(f"Generated {num_ensemble} ensemble structures for {seq_name}")


def train(fabric, model, optimizer1, optimizer2, batch, 
          ckpt_path=None,
          num_runs_phase_1=3,
          num_iterations_phase1=100, 
          num_iterations_phase2=500,
          early_stopping=True,
          intermediate_output_path=None,):
    
    # setup training
    model.train()
    best_loss = float('inf')
    intermediate_pdb_path = Path(f'{intermediate_output_path}/{Path(ckpt_path).stem}') if intermediate_output_path is not None else None
    ckpt_path_phase_1 = ckpt_path.replace('.ckpt', '_phase1.ckpt')  
    ckpt_path_phase_2 = ckpt_path.replace('.ckpt', '_phase2.ckpt') 

    # phase 1 training
    for r in range(num_runs_phase_1):
        model.initialize_parameters(batch['msa_feat'])
        for i in tqdm(range(num_iterations_phase1)):
            
            # clear gradientsj
            optimizer1.zero_grad()

            # create new copy of a batch for each iteration
            iteration_batch = {k: v for k, v in batch.items()}

            # forward pass
            outputs = model(iteration_batch)

            # calculate loss
            batch_no_recycling = tensor_tree_map(lambda t: t[0, ..., -1], batch)  # remove recycling dimension
            loss = model.loss(outputs, batch_no_recycling) 

            # backwards pass and update weights
            fabric.backward(loss)
            optimizer1.step()

            if intermediate_pdb_path is not None:
                fabric.log(f"loss/{intermediate_pdb_path.stem}_phase1", loss)
                pdb_path_output = f'{intermediate_pdb_path}_phase1_run_{r}_iter_{i}.pdb'
                save_intermediate_optimization_steps({**outputs, **batch_no_recycling}, pdb_path_output)
            else:
                fabric.log("loss/phase1", loss)

        # save checkpoint if best so far
        if loss < best_loss and ckpt_path_phase_1 is not None:
            best_loss = loss
            state = {"model": model, "optimizer1": optimizer1, "optimizer2": optimizer2, "iter": i}
            fabric.save(ckpt_path_phase_1, state)

    # load checkpoint with best outcome
    fabric.load(ckpt_path_phase_1, state)

    # phase 2 training
    no_improvement_count = 0
    for i in tqdm(range(num_iterations_phase2)):        
        # clear gradients
        optimizer2.zero_grad()

        # create new copy of a batch for each iteration
        iteration_batch = {k: v for k, v in batch.items()}

        # forward pass
        outputs = model(iteration_batch)

        # calculate loss
        batch_no_recycling = tensor_tree_map(lambda t: t[0, ..., -1], batch)
        loss = model.loss(outputs, batch_no_recycling) 

        # backwards pass and update weights
        fabric.backward(loss)
        optimizer2.step()

        # early stopping check
        if early_stopping:
            min_delta = 0.1
            patience = 50
            if loss < best_loss - min_delta:
                best_loss = loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    break
        
        # save intermediate output check
        if intermediate_pdb_path is not None:
            fabric.log(f"loss/{intermediate_pdb_path.stem}_phase2", loss)
            pdb_path_output = f'{intermediate_pdb_path}_phase2_iter_{i}.pdb'
            save_intermediate_optimization_steps({**outputs, **batch_no_recycling}, pdb_path_output)
        else:
            fabric.log("loss/phase2", loss)
    
    state = {"model": model, "optimizer1": optimizer1, "optimizer2": optimizer2, "iter": i}
    fabric.save(ckpt_path_phase_2, state)
    
    return loss


def save_intermediate_optimization_steps(outputs, path):
    # copy and detach relevant tensors
    out_to_prot_keys = ['final_atom_positions', 'plddt', 'atom37_atom_exists', 'aatype', 'residue_index', 'chain_index']
    output_info = {k: v.clone().detach() for k, v in outputs.items() if k in out_to_prot_keys}
    unrelaxed_protein = output_to_protein(output_info)

    # save intermediate output
    with open(path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))