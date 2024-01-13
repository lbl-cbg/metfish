from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.PDBParser import PDBParser
import numpy as np
from periodictable import elements
from scipy.spatial.distance import pdist, squareform

n_elec_df = pd.Series({el.symbol: el.number for el in elements})

def get_Pr(pdb_path):
    model = PDBParser().get_structure("", pdb_path)

    # Get atomic coordinates and atomic weights i.e. # of electrons
    coords = list()
    weights = list()
    for i, res in enumerate(model.get_residues()):
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

    hist, bins = np.histogram(distances, bins=100, density=True, weights=dist_weights)
