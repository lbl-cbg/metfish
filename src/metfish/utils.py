import os

from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.PDBParser import PDBParser
import numpy as np
from periodictable import elements
from scipy.spatial.distance import pdist, squareform


n_elec_df = {el.symbol: el.number for el in elements}


def get_Pr(structure, structure_id="", dmax=None, step=0.5):
    """
    Args:
        stucture (Structure or str) : The BioPython structure or the path to
                                      the PDB or mmCIF to calculate the P(r)
                                      curve for.
        structure_id (str)          : If *structure* is a file path, the ID
                                      of the structure in the file to use.
                                      By default, assume one structure in
                                      the file
        dmax (int, float, None)     : the max distance between atoms to
                                      consider. default = None i.e. determine
                                      from max distance found in structure
        step (float)                : the bin width to use for building
                                      histogram. default = 0.5

    Returns:
        (r, p) : a tuple with *r* as the first element and *P(r)* as the
                 second element
    """
    if isinstance(structure, str):
        _, ext = os.path.splitext(structure)
        if ext == ".cif":
            structure = FastMMCIFParser().get_structure(structure_id, structure)
        elif ext == ".pdb":
            structure = PDBParser().get_structure(structure_id, structure)
        else:
            warnings.warn(f"Unrecognized extension '{ext}'. Attempting to read as a PDB file")
            structure = PDBParser().get_structure(structure_id, structure)

    # Get atomic coordinates and atomic weights i.e. number of electrons
    coords = list()
    weights = list()
    for i, res in enumerate(structure.get_residues()):
        if res.id[0] == " ":
            for i, atom in enumerate(res.get_atoms()):
                c = atom.coord
                elem = atom.element
                coords.append(atom.coord)
                weights.append(n_elec_df[elem])
    coords = np.array(coords)
    weights = np.array(weights)

    # Calculate distances between each atom
    distances = pdist(coords)

    # Calculate a weight matrix
    dist_weights = np.outer(weights, weights)
    np.fill_diagonal(dist_weights, 0)
    dist_weights = squareform(dist_weights)

    # Calculate histogram
    if dmax is None:
        dmax = distances.max()
        dmax = np.ceil(dmax / step) * step
    hist, r = np.histogram(distances, bins=np.arange(0, dmax + 0.1, step), weights=dist_weights)
    p = np.concatenate(([0], hist / hist.sum()))

    return r, p
