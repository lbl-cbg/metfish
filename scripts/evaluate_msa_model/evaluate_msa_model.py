
import itertools
import pandas as pd

from pathlib import Path
from metfish.msa_model.predict import inference
from openfold.np import protein

from biopandas.pdb import PandasPdb

from metfish.utils import get_rmsd, get_lddt, save_aligned_pdb, get_Pr

# steps of the data process
prep_testing_data = False  # prep testing data
run_inference = True # run AF and MSA saxs models
compare_outputs = False  # compare outputs

## set up data paths and configuration
data_dir = "/global/cfs/cdirs/m3513/metfish/apo_holo_data"
output_dir = "/pscratch/sd/s/smprince/projects/metfish/model_evaluation/msa_saxs_model/unfrozen_af_weights" # "/pscratch/sd/s/smprince/projects/metfish/model_evaluation"
ckpt_path =  "/pscratch/sd/s/smprince/projects/metfish/model_outputs/checkpoints/msa_saxs_model_unfrozen_ckpts/epoch=22-step=12000.ckpt" # "/pscratch/sd/s/smprince/projects/metfish/model_outputs/checkpoints/msa_saxs_model_ckpts/epoch=36-step=19000.ckpt"
apo_holo_data_csv = "/global/cfs/cdirs/m3513/metfish/apo_holo_data/input.csv"  # use subset for now, will use full later
training_data_csv = "/global/cfs/cdirs/m3513/metfish/PDB70_verB_fixed_data/result/msa/input_training.csv" 

if prep_testing_data:
    # check which apo / holo pairs were in the original training set
    apo_holo_df = pd.read_csv(apo_holo_data_csv)
    training_df = pd.read_csv(training_data_csv)

    # generate list of sequences to use as input (ones not in the training dataset)
    input_sequences_excl_training = [n for n in apo_holo_df['name'].tolist() if n not in training_df['name'].tolist()]

    # remove sequences that can't be parsed from the pdb string or if msa file doesn't exist
    input_sequences = []
    for seq_name in input_sequences_excl_training:
        try:
            pdbpath = f"{data_dir}/pdbs/{seq_name}_atom_only.pdb"
            with open(pdbpath, 'r') as f:
                pdb_str = f.read()
                protein.from_pdb_string(pdb_str, None)
        except ValueError:
            continue
    
        filepath = Path(f"{data_dir}/msa/{seq_name}/a3m/{seq_name}.a3m")
        if not filepath.is_file():
            continue
    
        input_sequences.append(seq_name)

    # save the outputs to a csv
    apo_holo_df.query('name in @input_sequences').to_csv(f"{data_dir}/input_no_training_data.csv", index=False)

    # calculate saxs curves for the input sequences
    for name in input_sequences:
        pdb_path = f"{data_dir}/pdbs/{name}_atom_only.pdb"
        r, p = get_Pr(pdb_path, name, None, 0.5)
        out = f"{data_dir}/saxs_r/{name}_atom_only.csv"
        pd.DataFrame({"r": r, "P(r)": p}).to_csv(out, index=False)

## run inference
if run_inference:
    # get outputs for models using SAXS curves
    inference(data_dir=data_dir,
            output_dir=output_dir,
            ckpt_path=ckpt_path,
            model_name='MSASAXS')

    # get outputs for models using plain AF
    inference(data_dir=data_dir,
            output_dir=output_dir,
            model_name='AlphaFold',
            original_weights=True)


## Compare outputs
# if compare_outputs:
    # setup data dirs
    shared_dir = "/global/cfs/cdirs/m3513/metfish"
    output_dir = "/global/cfs/cdirs/m3513/metfish/model_evaluation/msa_saxs_model/unfrozen_af_weights" # "/pscratch/sd/s/smprince/projects/metfish/model_evaluation"  
    
    # load data dir and apo/holo pair information
    test_data_df = pd.read_csv(f"{shared_dir}/apo_holo_data/input_no_training_data.csv")
    apo_holo_df = pd.read_csv(f"{shared_dir}/apo_holo_data/apo_holo_pairs.csv")

    pairs = list(zip(apo_holo_df['apo_id'], apo_holo_df['holo_id']))
    names = test_data_df['name'].to_list()

    def create_model_comparison_df(pairs, names, comparisons):
        # compile information for each apo/holo pair into a df
        data = []
        for name in names:
            # get pair info, skip if one of the pairs was in the training dataset
            name_alt = [(set(p) - {name}).pop() for p in pairs if name in p][0] 
            if name_alt not in test_data_df['name'].to_list():
                continue
            
            # load apo holo data
            rmsd_apo_holo = apo_holo_df.query('apo_id == @name | holo_id == @name')['rmsd_apo_holo'].values[0]

            # load alignment data
            fnames = dict(true=f"{shared_dir}/apo_holo_data/pdbs/{name}_atom_only.pdb",
                        true_alt=f"{shared_dir}/apo_holo_data/pdbs/{name_alt}_atom_only.pdb",
                        out=f"{output_dir}/{name}_MSASAXS_unrelaxed.pdb",
                        out_alt=f"{output_dir}/{name_alt}_MSASAXS_unrelaxed.pdb",
                        out_af=f"{output_dir}/{name}_AlphaFold_unrelaxed.pdb",)
            for (a, b) in comparisons:
                # load pdb data
                pdb_df_a = PandasPdb().read_pdb(fnames[a]).df['ATOM']
                pdb_df_b = PandasPdb().read_pdb(fnames[b]).df['ATOM']

                # get plddt values
                plddt_res_num_a = pdb_df_a.drop_duplicates('residue_number')['residue_number'].to_numpy()
                plddt_res_num_b = pdb_df_b.drop_duplicates('residue_number')['residue_number'].to_numpy()
                plddt_a = pdb_df_a.drop_duplicates('residue_number')['b_factor'].to_numpy()
                plddt_b = pdb_df_b.drop_duplicates('residue_number')['b_factor'].to_numpy()
                plddt_a = 100 - plddt_a if (a == 'true' or a == 'true_alt') else plddt_a  # if from pdb, convert b_factors to plddt
                plddt_b = 100 - plddt_b if (b == 'true' or b == 'true_alt') else plddt_b  # if from pdb, convert b_factors to plddt

                # calculate saxs curves
                r_a, p_of_r_a = get_Pr(fnames[a], name, None, 0.5)
                r_b, p_of_r_b = get_Pr(fnames[b], name, None, 0.5)

                # run comparisons / alignments[
                comparison = f'{a}_vs_{b}'
                rmsd = get_rmsd(pdb_df_a, pdb_df_b)
                lddt = get_lddt(pdb_df_a, pdb_df_b)
                save_aligned_pdb(fnames[a], fnames[b], comparison)

                # add comparisons
                data.append(dict(name=name,
                                name_alt=name_alt,
                                type_a=a,
                                type_b=b,
                                comparison=comparison,
                                rmsd=rmsd,
                                lddt=lddt,
                                plddt_a=plddt_a,
                                plddt_bins_a=plddt_res_num_a,
                                plddt_b=plddt_b,
                                plddt_bins_b=plddt_res_num_b,
                                saxs_bins_a=r_a,
                                saxs_a=p_of_r_a,
                                saxs_bins_b=r_b,
                                saxs_b=p_of_r_b,
                                rmsd_apo_holo=rmsd_apo_holo,
                                fname_a=fnames[a],
                                fname_b=fnames[b],))

        return pd.DataFrame(data)

    # compare alphafold outputs with msa_saxs outputs
    types = ['out', 'true', 'out_af']
    comparisons = list(itertools.combinations(types, 2))

    df_af_saxs = create_model_comparison_df(pairs, names, comparisons)
    df_af_saxs.to_csv(f"{output_dir}/model_comparison_apo_vs_saxs.csv", index=False)

    # compare apo vs. holo data
    comparisons = [('out', 'true'), ('out_alt', 'true_alt'),('out', 'out_alt')]
    df_apo_holo = create_model_comparison_df(pairs, names, comparisons)
    df_apo_holo.to_csv(f"{output_dir}/model_comparison_apo_vs_holo.csv", index=False)

print('done')