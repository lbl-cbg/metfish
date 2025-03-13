import torch
from Bio.PDB import PDBParser
from pathlib import Path

from metfish.msa_model.utils.loss import differentiable_histogram, compute_saxs
from metfish.utils import get_Pr
from metfish.msa_model.data.data_pipeline import DataPipeline

def test_differentiable_histogram():
    values = torch.tensor([0.5, 1.5, 2.5])
    weights = torch.tensor([1.0, 2.0, 1.0])
    bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0])
    hist = differentiable_histogram(values, weights, bin_edges)
    
    expected_hist = torch.tensor([1.0, 2.0, 1.0])
    assert torch.allclose(hist, expected_hist, atol=1e-1), f"Expected {expected_hist}, but got {hist}"

def test_compute_saxs_vs_get_pr():
    # Use get_pr to calculate saxs profile
    step = 0.5
    dmax = 256
    
    pdb_path = Path("/pscratch/sd/s/smprince/projects/metfish_v2/metfish/tests/data/3DB7_A.pdb")
    structure = PDBParser().get_structure("", pdb_path)
    bins, expected_hist = get_Pr(structure, step=step, dmax=dmax)
    expected_hist = torch.tensor(expected_hist, dtype=torch.float32)

    # Use compute_saxs to calculate saxs profile
    data_pipeline = DataPipeline(template_featurizer=None)

    pdb_features = data_pipeline.process_pdb_feats(str(pdb_path), is_distillation=False)
    all_atom_pos = torch.tensor(pdb_features["all_atom_positions"], dtype=torch.float32)
    all_atom_mask = torch.tensor(pdb_features["all_atom_mask"], dtype=torch.float32)
    hist = compute_saxs(all_atom_pos, all_atom_mask, step=step, dmax=dmax)

    assert torch.allclose(hist, expected_hist, atol=1e-2), f"Expected {expected_hist}, but got {hist}"

def test_compute_saxs_grad():
    all_atom_pos = torch.randn(1, 37, 3, requires_grad=True)
    all_atom_mask = torch.ones(1, 37)
    saxs_profile = compute_saxs(all_atom_pos, all_atom_mask)
    
    assert saxs_profile.requires_grad, "Output tensor does not have grad_fn"
