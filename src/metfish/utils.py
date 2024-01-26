import os
import warnings

from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.PDBParser import PDBParser
import numpy as np
from periodictable import elements
from scipy.spatial.distance import pdist, squareform
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


n_elec_df = {el.symbol: el.number for el in elements}


def get_Pr(structure, structure_id="", dmax=None, step=0.5):
    """
    Args:
        structure (Structure or str) : The BioPython structure or the path to
                                       the PDB or mmCIF to calculate the P(r)
                                       curve for.
        structure_id (str)           : If *structure* is a file path, the ID
                                       of the structure in the file to use.
                                       By default, assume one structure in
                                       the file
        dmax (int, float, None)      : the max distance between atoms to
                                       consider. default = None i.e. determine
                                       from max distance found in structure
        step (float)                 : the bin width to use for building
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

def extract_seq(pdb_input,output_path):
    """
    Args:
        pdb_input   : The path to the PDB to extract sequence.
                      The PDB must only contain a single chain. 
                      pdbfixer is a good tool to prepare PDB for this function
        output_path : The path to store the output fasta file.
                      Example: /location/to/store/PDB.fasta
    Returns:
        There is no return for this function. The sequence will be written 
        as a fasta file in the give location.
    """
    pdb_name=os.path.basename(pdb_input).split(".")[0]
    counter=1
    for record in SeqIO.parse(pdb_input,"pdb-atom"):
        if counter > 1:
            raise ValueError("More than 1 Chain is in the file {}".format(pdb_input))
        else:
            new_seq_record = SeqRecord(record.seq, id=pdb_name, description='')
            SeqIO.write(new_seq_record, output_path ,"fasta")
        counter+=1