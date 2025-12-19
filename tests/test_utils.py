"""
Tests for core utility functions in metfish.utils
"""
import pytest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from Bio.PDB import PDBParser
from biopandas.pdb import PandasPdb

from metfish.utils import (
    get_Pr,
    extract_seq,
    get_single_letter_sequences,
    align_sequences,
    clean_pdb_df,
    superimpose_structures,
    get_rmsd,
    get_per_residue_rmsd,
    get_lddt,
    lddt,
    sample_conformers,
    write_conformers,
)


class TestGetPr:
    """Tests for get_Pr function."""
    
    @pytest.mark.requires_data
    def test_get_pr_from_pdb(self, test_data_dir):
        """Test P(r) calculation from PDB file."""
        pdb_path = test_data_dir / "3nir.pdb"
        r, p = get_Pr(str(pdb_path))
        
        assert len(r) == len(p)
        assert len(r) > 0
        assert np.all(r >= 0)
        assert np.all(p >= 0)
        assert np.isclose(p.sum(), 1.0, rtol=0.01)  # Should be normalized
    
    @pytest.mark.requires_data
    def test_get_pr_from_cif(self, test_data_dir):
        """Test P(r) calculation from CIF file."""
        cif_path = test_data_dir / "3nir.cif"
        r, p = get_Pr(str(cif_path))
        
        assert len(r) == len(p)
        assert len(r) > 0
        assert np.all(r >= 0)
        assert np.all(p >= 0)
    
    @pytest.mark.requires_data
    def test_get_pr_from_structure(self, sample_structure):
        """Test P(r) calculation from BioPython structure."""
        r, p = get_Pr(sample_structure)
        
        assert len(r) == len(p)
        assert len(r) > 0
    
    @pytest.mark.requires_data
    def test_get_pr_with_dmax(self, sample_structure):
        """Test P(r) calculation with specified dmax."""
        dmax = 50.0
        step = 0.5
        r, p = get_Pr(sample_structure, dmax=dmax, step=step)
        
        assert r.max() <= dmax
        assert np.isclose(r[1] - r[0], step)
    
    @pytest.mark.requires_data
    def test_get_pr_with_custom_step(self, sample_structure):
        """Test P(r) calculation with custom step size."""
        step = 1.0
        r, p = get_Pr(sample_structure, step=step)
        
        assert np.isclose(r[1] - r[0], step)
    
    @pytest.mark.requires_data
    def test_get_pr_consistency(self, test_data_dir):
        """Test that P(r) is consistent between PDB and CIF."""
        pdb_path = test_data_dir / "3nir.pdb"
        cif_path = test_data_dir / "3nir.cif"
        
        r_pdb, p_pdb = get_Pr(str(pdb_path), step=0.5)
        r_cif, p_cif = get_Pr(str(cif_path), step=0.5)
        
        nptest.assert_allclose(r_pdb, r_cif, rtol=1e-3)
        nptest.assert_allclose(p_pdb, p_cif, rtol=1e-2)


class TestExtractSeq:
    """Tests for extract_seq function."""
    
    @pytest.mark.requires_data
    def test_extract_seq_basic(self, test_data_dir, temp_dir):
        """Test basic sequence extraction."""
        pdb_path = test_data_dir / "3nir.pdb"
        output_path = temp_dir / "test_output.fasta"
        
        extract_seq(str(pdb_path), str(output_path))
        
        assert output_path.exists()
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) >= 2
        assert lines[0].startswith('>')
    
    @pytest.mark.requires_data
    def test_extract_seq_with_gaps(self, test_data_dir, temp_dir):
        """Test sequence extraction with residue gaps."""
        pdb_path = test_data_dir / "3DB7_A.pdb"
        output_path = temp_dir / "test_output.fasta"
        
        extract_seq(str(pdb_path), str(output_path))
        
        with open(output_path, 'r') as f:
            created_seq = f.readlines()[1].strip()
        
        with open(test_data_dir / "3DB7_A.fasta", 'r') as f:
            reference_seq = f.readlines()[1].strip()
        
        assert created_seq == reference_seq
    
    @pytest.mark.requires_data
    def test_extract_seq_multiple_chains_error(self, test_data_dir):
        """Test that multiple chains raise an error."""
        pdb_path = test_data_dir / "5K1S_1.pdb"
        
        with pytest.raises(ValueError, match="More than 1 Chain"):
            extract_seq(str(pdb_path), "output.fasta")


class TestSequenceAlignment:
    """Tests for sequence alignment functions."""
    
    @pytest.mark.requires_data
    def test_get_single_letter_sequences(self):
        """Test conversion to single letter sequences."""
        residues = ['ALA', 'GLY', 'VAL']
        seq = get_single_letter_sequences(residues)
        
        assert seq == 'AGV'
    
    @pytest.mark.requires_data
    def test_align_sequences_identical(self, sample_pdb_df):
        """Test alignment of identical sequences."""
        df1 = sample_pdb_df.copy()
        df2 = sample_pdb_df.copy()
        
        aligned_df1, aligned_df2 = align_sequences(df1, df2)
        
        assert len(aligned_df1) == len(aligned_df2)
    
    @pytest.mark.requires_data
    def test_clean_pdb_df(self, sample_pdb_df):
        """Test PDB dataframe cleaning."""
        atom_types = ['CA', 'N', 'C', 'O']
        cleaned_df = clean_pdb_df(sample_pdb_df, atom_types)
        
        assert set(cleaned_df['atom_name'].unique()).issubset(set(atom_types))


class TestStructureComparison:
    """Tests for structure comparison functions."""
    
    @pytest.mark.requires_data
    def test_superimpose_structures(self, test_data_dir):
        """Test structure superimposition."""
        pdb_path = test_data_dir / "3nir.pdb"
        df1 = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        df2 = df1.copy()
        
        si = superimpose_structures(df1, df2)
        
        assert si is not None
        assert hasattr(si, 'get_rms')
    
    @pytest.mark.requires_data
    def test_get_rmsd_identical(self, test_data_dir):
        """Test RMSD calculation for identical structures."""
        pdb_path = test_data_dir / "3nir.pdb"
        df1 = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        df2 = df1.copy()
        
        rmsd = get_rmsd(df1, df2)
        
        assert rmsd < 1e-3  # Should be very small for identical structures
    
    @pytest.mark.requires_data
    def test_get_per_residue_rmsd(self, test_data_dir):
        """Test per-residue RMSD calculation."""
        pdb_path = test_data_dir / "3nir.pdb"
        df1 = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        df2 = df1.copy()
        
        per_res_rmsd = get_per_residue_rmsd(df1, df2)
        
        assert len(per_res_rmsd) > 0
        assert all(isinstance(r, np.ndarray) for r in per_res_rmsd)


class TestLDDT:
    """Tests for lDDT calculation."""
    
    def test_lddt_identical_structures(self):
        """Test lDDT for identical structures."""
        coords = np.random.rand(1, 10, 3)
        mask = np.ones((1, 10, 1))
        
        score = lddt(coords, coords, mask)
        
        assert np.isclose(score, 1.0)
    
    def test_lddt_different_structures(self):
        """Test lDDT for different structures."""
        coords1 = np.random.rand(1, 10, 3)
        coords2 = np.random.rand(1, 10, 3)
        mask = np.ones((1, 10, 1))
        
        score = lddt(coords1, coords2, mask)
        
        assert 0.0 <= score <= 1.0
    
    def test_lddt_with_mask(self):
        """Test lDDT with partial masking."""
        coords1 = np.random.rand(1, 10, 3)
        coords2 = coords1.copy()
        mask = np.ones((1, 10, 1))
        mask[0, 5:, 0] = 0  # Mask out last 5 residues
        
        score = lddt(coords1, coords2, mask)
        
        assert np.isclose(score, 1.0)
    
    @pytest.mark.requires_data
    def test_get_lddt_with_dataframes(self, test_data_dir):
        """Test get_lddt with PDB dataframes."""
        pdb_path = test_data_dir / "3nir.pdb"
        df1 = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        df2 = df1.copy()
        
        score = get_lddt(df1, df2)
        
        assert score is not None
        assert 0.0 <= score <= 1.0


class TestConformerGeneration:
    """Tests for conformer generation functions."""
    
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_sample_conformers_anm(self, test_data_dir):
        """Test conformer sampling with ANM."""
        pdb_path = test_data_dir / "3nir.pdb"
        
        protein = sample_conformers(
            str(pdb_path),
            n_modes=1,
            n_confs=2,
            rmsd=1.0,
            type='ANM'
        )
        
        assert protein is not None
        assert protein.numCoordsets() > 1
    
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_sample_conformers_gnm(self, test_data_dir):
        """Test conformer sampling with GNM."""
        pdb_path = test_data_dir / "3nir.pdb"
        
        protein = sample_conformers(
            str(pdb_path),
            n_modes=1,
            n_confs=2,
            rmsd=1.0,
            type='GNM'
        )
        
        assert protein is not None
        assert protein.numCoordsets() > 1
    
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_write_conformers(self, test_data_dir, temp_dir):
        """Test writing conformers to files."""
        pdb_path = test_data_dir / "3nir.pdb"
        
        protein = sample_conformers(
            str(pdb_path),
            n_modes=1,
            n_confs=2,
            rmsd=1.0
        )
        
        filenames = write_conformers(temp_dir, "test", protein)
        
        assert len(filenames) > 0
        assert all(Path(f).exists() for f in filenames)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_get_pr_empty_structure(self):
        """Test get_Pr with invalid structure."""
        with pytest.raises(Exception):
            get_Pr("nonexistent_file.pdb")
    
    def test_extract_seq_invalid_path(self):
        """Test extract_seq with invalid path."""
        with pytest.raises(Exception):
            extract_seq("nonexistent.pdb", "output.fasta")
    
    def test_lddt_mismatched_shapes(self):
        """Test lDDT with mismatched coordinate shapes."""
        coords1 = np.random.rand(1, 10, 3)
        coords2 = np.random.rand(1, 5, 3)
        mask = np.ones((1, 10, 1))
        
        # Should handle or raise appropriate error
        try:
            score = lddt(coords1, coords2, mask)
        except (ValueError, IndexError):
            pass  # Expected behavior