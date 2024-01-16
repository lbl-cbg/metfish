from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.PDBParser import PDBParser
import numpy as np
from periodictable import elements
from scipy.spatial.distance import pdist, squareform


n_elec_df = pd.Series({el.symbol: el.number for el in elements})


def get_Pr(pdb_path):
    """
    Args:
        stucture (Structure or str) : The BioPython structure or the path to
                                      the PDB or mmCIF to calculate the P(r)
                                      curve for.
        structure_id (str)          : If *structure* is a file path, the ID
                                      of the structure in the file to use.
                                      By default, assume one structure in
                                      the file
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
    hist, bins = np.histogram(distances, bins=100, density=True, weights=dist_weights)

    return hist, bins
