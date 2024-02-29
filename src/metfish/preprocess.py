import pandas as pd
import pickle
import glob
import shutil

from metfish.utils import get_alphafold_atom_positions, get_rmsd, convert_pdb_to_sequence
from biopandas.pdb import PandasPdb
from pathlib import Path


def prep_conformer_pairs(data_dir, output_dir=None, n_pairs=6, af_output_dir=None):
    """Preprocess pdb files to generate fasta sequences, atom positions in AF structure format.

    Requires folder with the csv of apo/holo pairs and pdbs from Saldano et al., 2022

    Args:
        data_dir (str): the directory containing the pdb files
        output_dir (str, optional): the directory to save output files to. Defaults to data_dir if not provided
        n_pairs (int, optional): the # of pairs to look at, N pairs -> 2N structures predicted. Defaults to 6.
        af_output_dir (str, optional): path with alphafold outputs, used to calculate which conformer is less similar to the original AF prediction. Defaults to None.
    """
    # load apo and holo id names
    output_dir = output_dir or data_dir
    pairs_df = pd.read_csv(f"{data_dir}/apo_holo_pairs.csv")
    if n_pairs < pairs_df.shape[0]:
        pairs_df = pairs_df.sample(n_pairs, random_state=1234)  # selects random rows as subsample

    # convert pdb files into sequences, AF structures to use as AF inputs
    holo_id = pairs_df["holo_id"].to_list()
    apo_id = pairs_df["apo_id"].to_list()
    for name in [*holo_id, *apo_id]:
        # clean pdbs (extract only ATOM coordinates )
        raw_pdb = f"{data_dir}/{name}.pdb"
        clean_pdb = f"{output_dir}/pdbs/{name}_atom_only.pdb"
        Path(f"{output_dir}/pdbs/").mkdir(parents=True, exist_ok=True)
        PandasPdb().read_pdb(raw_pdb).to_pdb(clean_pdb, records=["ATOM"])

        # save pairs as fasta sequence files
        seq = convert_pdb_to_sequence(f"{output_dir}/pdbs/{name}_atom_only.pdb")
        seq = "\n".join([f">{name}", seq])
        Path(f"{output_dir}/sequences/apo_and_holo/").mkdir(parents=True, exist_ok=True)
        with open(f"{output_dir}/sequences/apo_and_holo/{name}.fasta", "w") as f:
            f.write(seq)

        # copy apo ids over to separate folder for AF input (AF only needs to run one of apo/holo pair bc same sequence)
        Path(f"{output_dir}/sequences/apo_only/").mkdir(parents=True, exist_ok=True)
        if name in apo_id:
            shutil.copyfile(
                f"{output_dir}/sequences/apo_and_holo/{name}.fasta", f"{output_dir}/sequences/apo_only/{name}.fasta"
            )

        # save pairs as alphafold structure representations
        struct = get_alphafold_atom_positions(f"{output_dir}/pdbs/{name}_atom_only.pdb")
        Path(f"{output_dir}/af_structures/").mkdir(parents=True, exist_ok=True)
        with open(f"{output_dir}/af_structures/{name}.pickle", "wb") as f:
            pickle.dump(struct, file=f)

    # calculate RMSD values between alphafold output and apo / holo conformers
    if af_output_dir is not None:
        rmsd_h, rmsd_a = list(), list()
        for h, a in zip(holo_id, apo_id):
            af_output = glob.glob(f"{af_output_dir}/{a}_unrelaxed_rank_001_*_000.pdb")[0]
            rmsd_h.append(get_rmsd(f"{output_dir}/pdbs/{h}_atom_only.pdb", af_output))  # same af output for same sequence
            rmsd_a.append(get_rmsd(f"{output_dir}/pdbs/{a}_atom_only.pdb", af_output))  # same af output for same sequence

        pairs_df["rmsd_apo_af"] = rmsd_a
        pairs_df["rmsd_holo_af"] = rmsd_h
        pairs_df["less_similar_conformer"] = pairs_df.apply(
            lambda x: x["holo_id"] if x["rmsd_apo_af"] < x["rmsd_holo_af"] else x["apo_id"], axis=1
        )

        pairs_df.to_csv(f"{output_dir}/apo_holo_pairs_with_similarity.csv", index=False)
