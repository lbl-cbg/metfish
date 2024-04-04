import os
import warnings
import numpy as np
import pandas as pd

from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio import SeqUtils, Align
from Bio.PDB.PDBIO import Select
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from periodictable import elements
from scipy.spatial.distance import pdist, squareform
from alphafold.common import protein
from alphafold.model import lddt


n_elec_df = {el.symbol: el.number for el in elements}
amino_acids = [a.upper() for a in SeqUtils.IUPACData.protein_letters_3to1.keys()]


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


def get_alphafold_atom_positions(fname: str):
    """ Use alphafold protein module to convert pdb file to alphafold's atomic position representation

    Args:
        fname (str): path to the pdb file

    Returns:
        structure: An np.ndarray with cartesian coordinates of atoms in angstroms [num_res, num_atom_type, 3]. 
        The atom types correspond to residue_constants.atom_types, i.e. the first three are N, CA, CB.
    """
    # read in file text and use af protein module to convert to position representation
    with open(fname, 'r') as file:
        pdb_str = file.read()
    prot = protein.from_pdb_string(pdb_str)  # protein module uses pdb file contents as input

    return prot.atom_positions

def save_clean_pdb(fname_original: str, fname_clean: str):
    """Rewrite pdb file with only ATOM entries"""
    raw_structure = PDBParser(QUIET=True).get_structure('', fname_original)

    class AtomsOnly(Select):
        def accept_residue(self, res):
            return res.get_id()[0] == " "  # this is the heteroatom field, if not empty is HETATM
        
    io = PDBIO()
    io.set_structure(raw_structure)
    io.save(fname_clean, select=AtomsOnly())

def convert_pdb_to_sequence(fname: str):
    """ Get single letter amino acid sequence from a pdb structure file """
    structure = PDBParser(QUIET=True).get_structure('', fname)
    residues = [res.resname for res in structure.get_residues() if res.resname in amino_acids]
    sequence = get_single_letter_sequences(residues)
    
    return sequence

def convert_pdb_to_atom_df(fname: str):
    """Get a data frame with atom coordinates, names, and residue names from a pdb structure file"""
    structure = PDBParser(QUIET=True).get_structure('', fname)
    
    atom_data = []
    for res in structure.get_residues():
        coords = [a.get_vector()[:] for a in res]
        types = [a.get_name() for a in res]
        atom_data.append(dict(residue_name=res.resname, atom_name=types, coords=coords))

    return pd.DataFrame(atom_data).explode(['atom_name', 'coords'])

def get_single_letter_sequences(residues):
    ret = list()
    for res in residues:
        res = res[0] + res[1:].lower()
        ret.append(SeqUtils.IUPACData.protein_letters_3to1[res])
    return "".join(ret)

def align_sequences(ref_df, query_df):
    """ Align protein sequences """
    ref_seq = get_single_letter_sequences(ref_df['residue_name'])
    query_seq = get_single_letter_sequences(query_df['residue_name'])
    
    # if not the same sequence, align
    if ref_seq != query_seq:
        aligner = Align.PairwiseAligner()
        alignments = aligner.align(ref_seq, query_seq)

        ref_idx, query_idx = alignments[0].indices[:, ~(alignments[0].indices == -1).any(axis=0)]
        
        ref_df = ref_df.iloc[ref_idx]
        query_df = query_df.iloc[query_idx]

    return ref_df, query_df

def superimpose_structures(fname_fixed, fname_moving, atom_types=["CA", "N", "C", "O"]):
    """ Superimpose two protein structures. 

    Args:
        fname_fixed (str): path to PDB file
        fname_moving (str): path to PDB file
        atom_types (list): atom types to align structures with, traditionally aligned with either
        1) only alpha-carbon atoms (CA), or 2) the "protein backbone" atoms (CA, N, C, O), or all atoms

    Returns:
        superimposer: returns instance of BioPython SVDSuperImposer class
    """

    # read in structures
    fixed_atom_df = convert_pdb_to_atom_df(fname_fixed)
    moving_atom_df = convert_pdb_to_atom_df(fname_moving)

    # filter for atom types and amino acide residues only
    fixed_atom_df = fixed_atom_df.query(f"residue_name in {amino_acids} & atom_name in {atom_types}")
    moving_atom_df = moving_atom_df.query(f"residue_name in {amino_acids} & atom_name in {atom_types}")

    # align sequences (if already aligned, will return same df)
    fixed_atom_df, moving_atom_df = align_sequences(fixed_atom_df, moving_atom_df)

    # get coordinates of the atoms
    fixed_coords = np.array(fixed_atom_df['coords'].to_list())
    moving_coords = np.array(moving_atom_df['coords'].to_list())

    # superimpose structures
    si = SVDSuperimposer()
    si.set(fixed_coords, moving_coords)
    si.run() # Run the SVD alignment
    
    return si

def get_rmsd(fname_a, fname_b, atom_types=["CA", "N", "C", "O"]):
    """ Calculate the RMSD between superimposed coordinates of two protein structures.
    """

    si = superimpose_structures(fname_a, fname_b, atom_types=atom_types)
    return si.get_rms()

def align_structures(fname_fixed, fname_moving):
    """Align two protein structures from pdb files and save aligned structures

    Args:
        fname_fixed (str): path to PDB file
        fname_moving (str): path to PDB file

    Returns:
        fname_aligned: path to aligned PDB file (rotate/translated version of fname_moving)
    """

    # load structure to transform
    structure = PDBParser(QUIET=True).get_structure('', fname_moving)

    # superimpose on experimental structure file 
    si = superimpose_structures(fname_fixed, fname_moving)
    rot, trans = si.get_rotran()
    for atom in structure.get_atoms():
        atom.transform(rot.astype("f"), trans.astype("f"))

    # save modified outputs as PDB files
    fname_aligned = f"{fname_moving.strip('.pdb')}_aligned.pdb"
    io = PDBIO()
    io.set_structure(structure) 
    io.save(fname_aligned)
    
    return fname_aligned

def get_lddt(fname_predicted, fname_true):
    structure_predicted = PDBParser(QUIET=True).get_structure('', fname_predicted)
    structure_true = PDBParser(QUIET=True).get_structure('', fname_true)
    coords_predicted = [a.get_vector()[:] for res in structure_predicted.get_residues() for a in res]
    coords_true = [a.get_vector()[:] for res in structure_true.get_residues() for a in res]

    coords_predicted = np.array(coords_predicted)[np.newaxis, :, :]
    coords_true = np.array(coords_true)[np.newaxis, :, :]
    true_pos_mask = np.array([[[1]] * np.shape(coords_true)[1]])

    return np.asarray(lddt.lddt(coords_predicted, coords_true, true_pos_mask))[0]
