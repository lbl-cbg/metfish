import pandas as pd
import pickle
import glob
import shutil

from metfish.utils import get_alphafold_atom_positions, get_rmsd, convert_pdb_to_sequence
from biopandas.pdb import PandasPdb

# set up preprocessing parameters and directories
calculate_less_similar = True  # calculates which conformer is less similar to the original AF prediction
use_subset = True  # only selects subset of proteins to run preprocessing
n_pairs = 6  # if using a subest, the # of pairs to look at, N pairs -> 2N structures predicted
data_dir = "metfish/data"  # directory with pdb files / csv from Saldano et al., 2022
output_dir = "metfish/data/output/no_manipulation/240213"  # directory with AF output files

# load apo and holo id names
pairs_df = pd.read_csv(f"{data_dir}/apo_holo_pairs.csv")
if use_subset:
    pairs_df = pairs_df.sample(n_pairs, random_state=1234)  # selects random rows as subsample

# convert pdb files into sequences, AF structures to use as AF inputs
holo_id = pairs_df["holo_id"].to_list()
apo_id = pairs_df["apo_id"].to_list()
for name in [*holo_id, *apo_id]:
    # clean pdbs (extract only ATOM coordinates )
    raw_pdb = f"{data_dir}/pdbs_raw/{name}.pdb"
    clean_pdb = f"{data_dir}/pdbs/{name}_atom_only.pdb"
    PandasPdb().read_pdb(raw_pdb).to_pdb(clean_pdb, records=["ATOM"])

    # save pairs as fasta sequence files
    seq = convert_pdb_to_sequence(f"{data_dir}/pdbs/{name}_atom_only.pdb")
    seq = "\n".join([f">{name}", seq])
    with open(f"{data_dir}/sequences/apo_and_holo/{name}.fasta", "w") as f:
        f.write(seq)

    # copy apo ids over to separate folder for AF input (AF only needs to run one of apo/holo pair bc same sequence)
    if name in apo_id:
        shutil.copyfile(
            f"{data_dir}/sequences/apo_and_holo/{name}.fasta", f"{data_dir}/sequences/apo_only/{name}.fasta"
        )

    # save pairs as alphafold structure representations
    struct = get_alphafold_atom_positions(f"{data_dir}/pdbs/{name}_atom_only.pdb")
    with open(f"{data_dir}/af_structures/{name}.pickle", "wb") as f:
        data = pickle.dump(struct, file=f)

# calculate RMSD values between alphafold output and apo / holo conformers
# NOTE - you need to run alphafold in advance to calculate these values
if calculate_less_similar:
    rmsd_h, rmsd_a = list(), list()
    for h, a in zip(holo_id, apo_id):
        af_output = glob.glob(f"{output_dir}/{a}_unrelaxed_rank_001_*_000.pdb")[0]
        rmsd_h.append(get_rmsd(f"{data_dir}/pdbs/{h}_atom_only.pdb", af_output))  # same af output for same sequence
        rmsd_a.append(get_rmsd(f"{data_dir}/pdbs/{a}_atom_only.pdb", af_output))  # same af output for same sequence

    pairs_df["rmsd_apo_af"] = rmsd_a
    pairs_df["rmsd_holo_af"] = rmsd_h
    pairs_df["less_similar_conformer"] = pairs_df.apply(
        lambda x: x["holo_id"] if x["rmsd_apo_af"] < x["rmsd_holo_af"] else x["apo_id"], axis=1
    )

    pairs_df.to_csv(f"{data_dir}/apo_holo_pairs_with_similarity.csv", index=False)

