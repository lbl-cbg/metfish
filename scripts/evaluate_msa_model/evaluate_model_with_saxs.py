
import pandas as pd
import shutil
import numpy as np

from scipy.special import rel_entr
from pathlib import Path
from metfish.msa_model.predict import inference
from openfold.np import protein

from biopandas.pdb import PandasPdb

from metfish.utils import get_rmsd, get_lddt, save_aligned_pdb, get_Pr, sample_conformers, write_conformers, get_per_residue_rmsd

# steps of the data process
prep_testing_data = False  # prep testing data
run_inference = True  # run AF and MSA saxs models
compare_outputs = True  # compare outputs
overwrite = True

## set up data paths and configuration
shared_dir = "/global/cfs/cdirs/m3513/metfish"
project_dir = "/pscratch/sd/s/smprince/projects/metfish"
ckpt_path =  f"{project_dir}/model_outputs/checkpoints/msa_saxs_model_with_saxs_loss_5_weight/epoch=18-step=19000.ckpt"
training_data_csv = f"{shared_dir}/PDB70_ANM_simulated_data/scripts/input_training.csv" 

apo_holo_dir = f"{shared_dir}/apo_holo_data"
apo_holo_data_csv = f"{shared_dir}/apo_holo_data/input.csv"  # use subset for now, will use full later
output_dir = f"{project_dir}/model_evaluation/model_with_saxs_loss/apoholo_data"

rmsd = 3
n_modes = 2
n_confs = 6
data_dir_nma = f"{apo_holo_dir}/nma_data"  # nma analysis performed on the apo holo dataset
nma_data_csv = "input_nma_data.csv"
output_dir_nma = f"/{project_dir}/model_evaluation/model_with_saxs_loss/nma_data"

def create_model_comparison_df(pairs, names, comparisons, data_dir, output_dir, pair_data_df, apo_holo_df):
    # compile information for each apo/holo pair into a df
    data = []
    for name in names:
        print(f'Generating comparison data for {name}...')
        # get pair info, skip if one of the pairs was in the training dataset
        name_alt = [(set(p) - {name}).pop() for p in pairs if name in p][0] 
        if name_alt not in pair_data_df['name'].to_list():
            continue
        if 'mode' in name_alt:
            name, name_alt = name_alt, name
        
        # load apo holo data
        rmsd_apo_holo = apo_holo_df.query('apo_id == @name | holo_id == @name')['rmsd_apo_holo']
        rmsd_apo_holo = rmsd_apo_holo.values[0] if not rmsd_apo_holo.empty else None

        # load alignment data
        fnames = dict(true=f"{data_dir}/pdbs/{name}_atom_only.pdb",
                      true_alt=f"{data_dir}/pdbs/{name_alt}_atom_only.pdb",
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

            # get saxs kl divergence
            eps = 1e-10
            saxs_a_padded = np.pad(p_of_r_a, (0, np.max([len(r_a), len(r_b)]) - len(r_a)), mode='constant', constant_values=0)
            saxs_b_padded = np.pad(p_of_r_b, (0, np.max([len(r_a), len(r_b)]) - len(r_b)), mode='constant', constant_values=0)
            saxs_a_padded = (saxs_a_padded + eps) / np.sum(saxs_a_padded + eps)
            saxs_b_padded = (saxs_b_padded + eps) / np.sum(saxs_b_padded + eps)
            saxs_kldiv = np.sum(rel_entr(saxs_a_padded, saxs_b_padded))

            # run comparisons / alignments[
            comparison = f'{a}_vs_{b}'
            rmsd = get_rmsd(pdb_df_a, pdb_df_b)
            lddt = get_lddt(pdb_df_a, pdb_df_b)
            per_residue_rmsd = get_per_residue_rmsd(pdb_df_a, pdb_df_b)
            save_aligned_pdb(fnames[a], fnames[b], comparison)

            # add comparisons
            data.append(dict(name=name,
                             name_alt=name_alt,
                            type_a=a,
                            type_b=b,
                            comparison=comparison,
                            rmsd=rmsd,
                            lddt=lddt,
                            saxs_kldiv=saxs_kldiv,
                            plddt_a=plddt_a,
                            plddt_bins_a=plddt_res_num_a,
                            plddt_b=plddt_b,
                            plddt_bins_b=plddt_res_num_b,
                            per_residue_rmsd=per_residue_rmsd,
                            saxs_bins_a=r_a,
                            saxs_a=p_of_r_a,
                            saxs_bins_b=r_b,
                            saxs_b=p_of_r_b,
                            rmsd_apo_holo=rmsd_apo_holo,
                            fname_a=fnames[a],
                            fname_b=fnames[b],))

    return pd.DataFrame(data)
    
if prep_testing_data:
    # check which apo / holo pairs were in the original training set
    apo_holo_df = pd.read_csv(apo_holo_data_csv)
    training_df = pd.read_csv(training_data_csv)

    # generate list of sequences to use as input (ones not in the training dataset)
    input_sequences_excl_training = [n for n in apo_holo_df['name'].tolist() if n not in training_df['name'].tolist()]

    # remove sequences that can't be parsed from the pdb string or if msa file doesn't exist
    input_sequences = []
    for seq_name in input_sequences_excl_training:
        with open(f"{apo_holo_dir}/sequences/{seq_name}.fasta", 'r') as file:
            lines = file.readlines()
            res = lines[1].strip()

        if len(res) <= 256:
            try:
                pdbpath = f"{apo_holo_dir}/pdbs/{seq_name}_atom_only.pdb"
                with open(pdbpath, 'r') as f:
                    pdb_str = f.read()
                    protein.from_pdb_string(pdb_str, None)
            except ValueError:
                continue
            except KeyError:
                continue
        
            filepath = Path(f"{apo_holo_dir}/msa/{seq_name}/a3m/{seq_name}.a3m")
            if not filepath.is_file():
                continue
        
            input_sequences.append(seq_name)

    # save the outputs to a csv
    apo_holo_df.query('name in @input_sequences').to_csv(f"{apo_holo_dir}/input_no_training_data.csv", index=False)

    # calculate saxs curves for the input sequences
    for name in input_sequences:
        pdb_path = f"{apo_holo_dir}/pdbs/{name}_atom_only.pdb"
        r, p = get_Pr(pdb_path, name, None, 0.5)
        out = f"{apo_holo_dir}/saxs_r/{name}_atom_only.csv"
        pd.DataFrame({"r": r, "P(r)": p}).to_csv(out, index=False)
    
    # prep nma outputs
    test_data_df = pd.read_csv(f"{apo_holo_dir}/input_no_training_data.csv")
    
    input_names = []
    input_sequences = []
    input_msa_ids = []
    for seq_name in test_data_df['name'].to_list():
        pdb_path = f"{apo_holo_dir}/pdbs/{seq_name}_atom_only.pdb"

        # generate simulated pdb files using normal mode analysis
        conformer_coords = sample_conformers(pdb_path, n_modes=n_modes, n_confs=n_confs, rmsd=rmsd)

        output_dir = f"{data_dir_nma}/pdbs"
        filenames = write_conformers(Path(output_dir), seq_name, conformer_coords, pdb_ext='_atom_only.pdb')

        # generate SAXS profiles from simulated PDB conformers
        for f in filenames:
            r, p = get_Pr(f, seq_name, None, 0.5)
            
            out = Path(data_dir_nma) / 'saxs_r' / f'{Path(f).stem}.csv'
            pd.DataFrame({"r": r, "P(r)": p}).to_csv(out, index=False)

        # copy msa path and rename
        original_msa_path = Path(f"{apo_holo_dir}/msa/{seq_name}/a3m/{seq_name}.a3m")
        new_msa_path = Path(f"{data_dir_nma}/msa/{seq_name}/a3m/{seq_name}.a3m")
        new_msa_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_msa_path, new_msa_path)

        # add relevant files to input data
        nma_names = [Path(f).stem.split('_atom_only')[0] for f in filenames if 'mode0' in f]
        sequence = test_data_df.query('name == @seq_name')['seqres'].to_list()
        input_names.extend(nma_names)
        input_msa_ids.extend([seq_name] * len(nma_names)) 
        input_sequences.extend(sequence * len(nma_names))

    # save the outputs to a csv
    (pd.DataFrame({"name": input_names, "seqres": input_sequences, "msa_id": input_msa_ids})
     .to_csv(f"{data_dir_nma}/{nma_data_csv}", index=False))
    

## run inference
if run_inference:
    # get outputs for models using SAXS curves
    inference(data_dir=apo_holo_dir,
            output_dir=output_dir,
            ckpt_path=ckpt_path,
            model_name='MSASAXS')

    # get outputs for models using plain AF
    inference(data_dir=apo_holo_dir,
            output_dir=output_dir,
            model_name='AlphaFold',
            original_weights=True)

    # get outputs for nma data
    inference(data_dir=data_dir_nma,
            output_dir=output_dir_nma,
            ckpt_path=ckpt_path,
            model_name='MSASAXS',
            test_csv_name=nma_data_csv,)

    inference(data_dir=data_dir_nma,
            output_dir=output_dir_nma,
            model_name='AlphaFold',
            original_weights=True,
            test_csv_name=nma_data_csv,)

## Compare outputs
if compare_outputs:
    # load data dir and apo/holo pair information
    test_data_df = pd.read_csv(f"{apo_holo_dir}/input_no_training_data.csv")
    apo_holo_df = pd.read_csv(f"{apo_holo_dir}/apo_holo_pairs.csv")

    pairs_apo_holo = list(zip(apo_holo_df['apo_id'], apo_holo_df['holo_id']))
    names = test_data_df['name'].to_list()
    
    if overwrite:
        types = ['out', 'true', 'out_af']
        comparisons = list(itertools.combinations(types, 2))

        df_af_saxs = create_model_comparison_df(pairs_apo_holo, names, comparisons, apo_holo_dir, output_dir, test_data_df, apo_holo_df)
        df_af_saxs.to_pickle(f'{output_dir}/model_comparison_apo_vs_saxs_local.pkl')
    else:
        df_af_saxs = pd.read_pickle(f'{output_dir}/model_comparison_apo_vs_saxs_local.pkl')

    if overwrite:
        comparisons = [('out', 'true'), ('out_alt', 'true_alt'),('out', 'out_alt')]
        df_apo_holo = create_model_comparison_df(pairs_apo_holo, names, comparisons, apo_holo_dir, output_dir, test_data_df, apo_holo_df)
        df_apo_holo.to_pickle(f'{output_dir}/model_comparison_apo_vs_holo_local.pkl')
    else:
        df_apo_holo = pd.read_pickle(f'{output_dir}/model_comparison_apo_vs_holo_local.pkl')

    # load data dir and apo/holo pair information
    if overwrite:
        nma_data_df = pd.read_csv(f"{data_dir_nma}/{nma_data_csv}")
        nma_names = nma_data_df['name'].to_list()
        
        pairs_modified = []
        for seq in nma_data_df['msa_id'].unique():
            pairs_modified.extend([(f'{seq}_mode0_conf0', f'{seq}_mode0_conf{i}') for i in range(n_confs + 1) if i != 0])

        df_nma = create_model_comparison_df(pairs_modified, nma_names, [('out', 'out_alt'), ('out', 'true_alt'), ('true', 'true_alt')], data_dir_nma, output_dir_nma, nma_data_df, apo_holo_df)
        df_nma.to_pickle(f'{output_dir_nma}/model_comparison_nma_local.pkl')
    else:
        df_nma = pd.read_pickle(f'{output_dir_nma}/model_comparison_nma_local.pkl')
