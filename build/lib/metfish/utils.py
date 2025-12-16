import os
import warnings
import numpy as np

from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.PDBParser import PDBParser
from biopandas.pdb import PandasPdb
from pathlib import Path
from periodictable import elements
from scipy.spatial.distance import pdist, squareform

from Bio.SVDSuperimposer import SVDSuperimposer
from Bio import SeqUtils, Align
from Bio.PDB import PDBIO

from prody import parsePDB, ANM, GNM, extendModel, traverseMode, writePDB

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

def extract_seq(pdb_input, output_path):
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
    seq_df = PandasPdb().read_pdb(pdb_input).amino3to1(record='ATOM', fillna='X')
    pdb_name = Path(pdb_input).stem

    if len(seq_df['chain_id'].unique()) > 1:
        raise ValueError(f"More than 1 Chain is in the file {pdb_input}")

    seq = ''.join(seq_df['residue_name'])
    seq = "\n".join([f">{pdb_name}", seq])
    with open(output_path, "w") as f:
        f.write(seq)

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

        alignment = alignments[0]
        
        ref_idx = []
        query_idx = []
        for ref_start, ref_end in alignment.aligned[0]:
            ref_idx.extend(range(ref_start, ref_end))
        for query_start, query_end in alignment.aligned[1]:
            query_idx.extend(range(query_start, query_end))
        
        ref_df = ref_df.iloc[ref_idx]
        query_df = query_df.iloc[query_idx]

    return ref_df, query_df

def clean_pdb_df(pdb_df, atom_types):
    alt_locs = pdb_df['alt_loc'].unique()
    alt_locs = alt_locs[:2] if len(alt_locs) > 1 else alt_locs

    pdb_df = pdb_df.query(f"residue_name in {amino_acids} & atom_name in {atom_types}")
    pdb_df = pdb_df[pdb_df["alt_loc"].isin(alt_locs)]  # query not work well with empty strings'

    return pdb_df

def superimpose_structures(fixed_df, moving_df, atom_types=["CA", "N", "C", "O"]):
    """ Superimpose two protein structures. 

    Args:
        fname_fixed (str): path to PDB file
        fname_moving (str): path to PDB file
        atom_types (list): atom types to align structures with, traditionally aligned with either
        1) only alpha-carbon atoms (CA), or 2) the "protein backbone" atoms (CA, N, C, O), or all atoms

    Returns:
        superimposer: returns instance of BioPython SVDSuperImposer class
    """

    # filter for atom types and amino acide residues only
    fixed_atom_df = clean_pdb_df(fixed_df, atom_types)
    moving_atom_df = clean_pdb_df(moving_df, atom_types)

    # align sequences (if already aligned, will return same df)
    fixed_atom_df, moving_atom_df = align_sequences(fixed_atom_df, moving_atom_df)

    # get coordinates of the atoms
    fixed_coords = fixed_atom_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    moving_coords = moving_atom_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()

    # superimpose structures
    si = SVDSuperimposer()
    si.set(fixed_coords, moving_coords)
    si.run() # Run the SVD alignment
    
    return si

def get_rmsd(fixed_atom_df, moving_atom_df, atom_types=["CA", "N", "C", "O"]):
    """ Calculate the RMSD between superimposed coordinates of two protein structures.
    """
    si = superimpose_structures(fixed_atom_df, moving_atom_df, atom_types=atom_types)
    return si.get_rms()

def get_per_residue_rmsd(fixed_atom_df, moving_atom_df, atom_types=["CA", "N", "C", "O"]):
    fixed_atom_df = clean_pdb_df(fixed_atom_df, atom_types)
    moving_atom_df = clean_pdb_df(moving_atom_df, atom_types)

    fixed_atom_df, moving_atom_df = align_sequences(fixed_atom_df, moving_atom_df)

    # get coordinates of the atoms
    fixed_coords = fixed_atom_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    moving_coords = moving_atom_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()

    # superimpose structures
    si = SVDSuperimposer()
    si.set(fixed_coords, moving_coords)
    si.run() # Run the SVD alignment

    fixed_coords = si.reference_coords
    moving_coords = si.get_transformed()
    
    assert len(fixed_coords) == len(fixed_atom_df)

    residue_ids = fixed_atom_df['residue_number'].unique()
    per_residue_rmsd = []
    for res_id in residue_ids:
        atoms_inds = fixed_atom_df.reset_index(drop=True).query(f"residue_number == {res_id}").index
        rmsd = np.sqrt(np.sum((fixed_coords[atoms_inds, :] - moving_coords[atoms_inds, :]) ** 2, axis=1))
        per_residue_rmsd.append(rmsd)

    return per_residue_rmsd


def get_lddt(predicted_atom_df, true_atom_df, atom_types=["CA", "N", "C", "O"]):
    predicted_atom_df = clean_pdb_df(predicted_atom_df, atom_types)
    true_atom_df = clean_pdb_df(true_atom_df, atom_types)

    coords_predicted = predicted_atom_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()[np.newaxis, :, :]
    coords_true = true_atom_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()[np.newaxis, :, :]
    true_pos_mask = np.array([[[1]] * np.shape(coords_true)[1]])

    if coords_predicted.shape[1] != coords_true.shape[1]:
        warnings.warn("The number of atoms in the predicted and true structures are different.", stacklevel=2)
        return None
    else:
        return np.asarray(lddt(coords_predicted, coords_true, true_pos_mask))[0]

def save_aligned_pdb(fname_fixed, fname_moving, fname_tag, output_dir=None):
    """Align two protein structures from pdb files and save aligned structures

    Args:
        fname_fixed (str): path to PDB file
        fname_moving (str): path to PDB file

    Returns:
        fname_aligned: path to aligned PDB file (rotate/translated version of fname_moving)
    """

    # load structure to transform
    structure = PDBParser(QUIET=True).get_structure('', fname_moving)
    df_fixed = PandasPdb().read_pdb(fname_fixed).df['ATOM']
    df_moving = PandasPdb().read_pdb(fname_moving).df['ATOM']

    # superimpose on experimental structure file 
    si = superimpose_structures(df_fixed, df_moving)
    rot, trans = si.get_rotran()
    for atom in structure.get_atoms():
        atom.transform(rot.astype("f"), trans.astype("f"))

    # save modified outputs as PDB files
    if output_dir is not None:
        fname_aligned = os.path.join(output_dir, f"{Path(fname_moving).stem}_aligned_{fname_tag}.pdb")
    else:
        fname_aligned = f"{fname_moving.rstrip('.pdb')}_aligned_{fname_tag}.pdb"
    
    io = PDBIO()
    io.set_structure(structure) 
    io.save(fname_aligned)
    
    return None

def lddt(predicted_points,
         true_points,
         true_points_mask,
         cutoff=15.,
         per_residue=False):
  """Measure (approximate) lDDT for a batch of coordinates.

  lDDT reference:
  Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
  superposition-free score for comparing protein structures and models using
  distance difference tests. Bioinformatics 29, 2722–2728 (2013).

  lDDT is a measure of the difference between the true distance matrix and the
  distance matrix of the predicted points.  The difference is computed only on
  points closer than cutoff *in the true structure*.

  This function does not compute the exact lDDT value that the original paper
  describes because it does not include terms for physical feasibility
  (e.g. bond length violations). Therefore this is only an approximate
  lDDT score.

  Args:
    predicted_points: (batch, length, 3) array of predicted 3D points
    true_points: (batch, length, 3) array of true 3D points
    true_points_mask: (batch, length, 1) binary-valued float array.  This mask
      should be 1 for points that exist in the true points.
    cutoff: Maximum distance for a pair of points to be included
    per_residue: If true, return score for each residue.  Note that the overall
      lDDT is not exactly the mean of the per_residue lDDT's because some
      residues have more contacts than others.

  Returns:
    An (approximate, see above) lDDT score in the range 0-1.
  """

  assert len(predicted_points.shape) == 3
  assert predicted_points.shape[-1] == 3
  assert true_points_mask.shape[-1] == 1
  assert len(true_points_mask.shape) == 3

  # Compute true and predicted distance matrices.
  dmat_true = np.sqrt(1e-10 + np.sum(
      (true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

  dmat_predicted = np.sqrt(1e-10 + np.sum(
      (predicted_points[:, :, None] -
       predicted_points[:, None, :])**2, axis=-1))

  dists_to_score = (
      (dmat_true < cutoff).astype(np.float32) * true_points_mask *
      np.transpose(true_points_mask, [0, 2, 1]) *
      (1. - np.eye(dmat_true.shape[1]))  # Exclude self-interaction.
  )

  # Shift unscored distances to be far away.
  dist_l1 = np.abs(dmat_true - dmat_predicted)

  # True lDDT uses a number of fixed bins.
  # We ignore the physical plausibility correction to lDDT, though.
  score = 0.25 * ((dist_l1 < 0.5).astype(np.float32) +
                  (dist_l1 < 1.0).astype(np.float32) +
                  (dist_l1 < 2.0).astype(np.float32) +
                  (dist_l1 < 4.0).astype(np.float32))

  # Normalize over the appropriate axes.
  reduce_axes = (-1,) if per_residue else (-2, -1)
  norm = 1. / (1e-10 + np.sum(dists_to_score, axis=reduce_axes))
  score = norm * (1e-10 + np.sum(dists_to_score * score, axis=reduce_axes))

  return score

# select alpha carbons
def sample_conformers(fname, n_modes=2, n_confs=6, rmsd=3.0, type='ANM'):
    protein = parsePDB(str(fname))
    calphas = protein.select('calpha')

    # perform normal mode analysis
    if type == 'ANM':
        anm = ANM('ANM analysis')
        anm.buildHessian(calphas)  # default values are cutoff=15.0 and gamma=1.0
        anm.calcModes(n_modes=n_modes, turbo=True)

        # extend the model
        nm_ext, _ = extendModel(anm, calphas, protein, norm=True)
    elif type == 'GNM':
        gnm = GNM('GNM analysis')
        gnm.buildKirchhoff(calphas)  # default values are cutoff=10.0 and gamma=1.0
        gnm.calcModes(n_modes=n_modes, turbo=True)

        # extend the model
        nm_ext, _ = extendModel(gnm, calphas, protein, norm=True)

    # sample conformations along the  nodes
    # rmsd = len(calphas) * rmsd_ratio  # 0.02 is a potential default value
    # ens = sampleModes(anm_ext, atoms=protein, n_confs=n_confs, rmsd=rmsd) # random sample (rmsd = avg)
    for i, mode in enumerate(nm_ext):
        ens = traverseMode(nm_ext[mode], atoms=protein, n_steps=int(n_confs/2), rmsd=rmsd)  # trajectory along a mode (rmsd = max)

        # write confirmations
        protein.addCoordset(ens.getCoordsets(), label=f'mode{i}')

    protein.all.setBetas(0)  # I believe these steps are mainly useful if later optimizations are applied
    protein.ca.setBetas(1)  

    return protein

def write_conformers(out_dir, name, protein, pdb_ext='.pdb'):
    # write the conformations to a pdb file
    filenames = []
    last_mode = None
    conf_idx = 0
    for i in range(protein.numCoordsets()):
        mode_label = protein.getCSLabels()[i]
        curr_mode = mode_label.replace("mode", "")
        if curr_mode != last_mode:
            last_mode = curr_mode
            conf_idx = 0
        
        if mode_label == "":
            filename = str(out_dir / f'{name}.pdb')
        else:
            filename = str(out_dir / f'{name}_{mode_label}_conf{conf_idx}{pdb_ext}')
        writePDB(filename, protein, csets=i)

        filenames.append(filename)
        conf_idx += 1 

    return filenames