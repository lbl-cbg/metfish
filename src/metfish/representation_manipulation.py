import pickle
import numpy as np
import pandas as pd
import warnings

from pathlib import Path


def modify_representations(prev: dict = None, method: str = "none", **kwargs):
    """Modify the pair, position (structural), or MSA (first row only) representations
    that will be used as inputs for the next recycling iteration of AlphaFold.

    Args:
        prev (dict): Dict of pair, position, and msa representations.
        method (str): Method to use to modify representations.

    Returns:
        dict: modified pair, position and msa representations
    """

    # apply modification method (test examples, will eventually modify prev outputs based on SAXS data)
    match method:
        case "none":
            repr_modified = prev
        case "reinitialize":
            repr_modified = reinitialize(prev)
        case "add_noise":
            repr_modified = add_noise(prev)
        case "replace_structure":
            repr_modified = replace_structure(prev, **kwargs)
        case _:
            warnings.warn("Representation modification method not supported - defaulting to no modification")
            repr_modified = prev

    return repr_modified


def add_noise(prev):
    """Adds gaussian noise to the pair, structure, and MSA representations"""
    prev_with_noise = dict()
    rng = np.random.default_rng()

    for key, value in prev.items():
        noise = rng.standard_normal(value.shape).astype("float16")  # gaussian noise with μ = 0, σ = 1
        prev_with_noise[key] = value + noise

    return prev_with_noise


def reinitialize(prev):
    """Reinitializes the pair, structure, and MSA, representations to zero arrays"""

    L = np.shape(prev["prev_pair"])[0]

    prev = {
        "prev_msa_first_row": np.zeros([L, 256], dtype=np.float16),
        "prev_pair": np.zeros([L, L, 128], dtype=np.float16),
        "prev_pos": np.zeros([L, 37, 3], dtype=np.float16),
    }

    return prev


def replace_structure(prev, job_name, input_dir=None, replacement_method="template"):
    """Replace intermediate structure (atom position) representations from alphafold"""
    # load in conformer pair information
    input_dir = input_dir or Path(__file__).resolve().parents[2] / 'data'
    conformer_pairs_fname = f"{input_dir}/apo_holo_pairs_with_similarity.csv"
    
    pdb_name = job_name.split("_")[0]
    conformer_df = pd.read_csv(conformer_pairs_fname)
    pairs = list(zip(conformer_df["apo_id"], conformer_df["holo_id"]))

    # get relevant pairs
    index = [ind for ind, (a, h) in enumerate(pairs) if pdb_name in a or pdb_name in h]
    pair_info = conformer_df.iloc[index, :]

    # get structure name depending on replacement method
    match replacement_method:
        case "less_similar":
            replacement_pdb_name = pair_info["less_similar_conformer"]
        case "alternate":
            replacement_pdb_name = (
                pair_info["holo_id"] if pdb_name in pair_info["apo_id"].to_list()[0] else pair_info["apo_id"]
            )  # get opposite conformer
        case "template":
            replacement_pdb_name = (
                pair_info["apo_id"] if pdb_name in pair_info["apo_id"].to_list()[0] else pair_info["holo_id"]
            )  # provide conformer experimental structure
        case _:
            replacement_pdb_name = []

    if any(replacement_pdb_name):
        # load replacement structure
        print(f"Replacing {pdb_name} intermediate structure with {replacement_pdb_name.to_list()[0]}.")
        with open(f"{input_dir}/af_structures/{replacement_pdb_name.to_list()[0]}.pickle", "rb") as f:
            replacement_structure = pickle.load(f)

        # NOTE - additional zero values seem to get added to the first dimension of the position array
        # (n_res) when running multiple sequences. AF ignores anything longer than n_res when writing
        # to a pdb file from the protein class, so replacing the first n_res values and adding a warning
        if np.shape(prev["prev_pos"])[0] != np.shape(replacement_structure)[0]:
            warnings.warn(
                f"Alphafold intermediate {np.shape(prev['prev_pos'])} and modified conformer "
                f"{np.shape(replacement_structure)} structures were not the same shape.",
            )

        # replace conformer with alternative option
        n_res = np.shape(replacement_structure)[0]
        prev["prev_pos"][:n_res, :, :] = replacement_structure.astype("float16")

    else:
        warnings.warn(f'No replacement option found for "{pdb_name}". Continuing without modification')

    return prev
