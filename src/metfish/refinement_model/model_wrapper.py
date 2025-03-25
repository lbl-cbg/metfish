
from tqdm import tqdm
from pathlib import Path
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import protein

from metfish.utils import output_to_protein


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

            # log the loss
            fabric.log("loss", loss)

            if intermediate_pdb_path is not None:
                pdb_path_output = f'{intermediate_pdb_path}_phase1_run_{r}_iter_{i}.pdb'
                save_intermediate_optimization_steps({**outputs, **batch_no_recycling}, pdb_path_output)
        
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

        # log the loss
        fabric.log("loss", loss)

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
            pdb_path_output = f'{intermediate_pdb_path}_phase2_iter_{i}.pdb'
            save_intermediate_optimization_steps({**outputs, **batch_no_recycling}, pdb_path_output)
    
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