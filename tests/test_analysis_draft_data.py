"""
Tests for analysis_draft.py - Data Processing
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch


class TestSharedUtilities:
    """Test shared utility functions."""
    
    def test_nm_to_angstrom_constant(self):
        """Test unit conversion constant."""
        from metfish.analysis.analysis_draft import NM_TO_ANGSTROM
        assert NM_TO_ANGSTROM == 10.0
    
    def test_default_contact_cutoff(self):
        """Test default contact cutoff constant."""
        from metfish.analysis.analysis_draft import DEFAULT_CONTACT_CUTOFF_NM
        assert DEFAULT_CONTACT_CUTOFF_NM == 0.8
    
    def test_l1_loss_padded_equal_length(self):
        """Test L1 loss with equal length arrays."""
        from metfish.analysis.analysis_draft import l1_loss_padded
        
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5, 3.5])
        result = l1_loss_padded(a, b)
        assert result == pytest.approx(1.5)
    
    def test_l1_loss_padded_unequal_length(self):
        """Test L1 loss with unequal length arrays."""
        from metfish.analysis.analysis_draft import l1_loss_padded
        
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        result = l1_loss_padded(a, b)
        assert result == pytest.approx(3.0)


class TestDataPaths:
    """Test DataPaths class."""
    
    def test_paths_are_path_objects(self):
        """Test that paths are Path objects."""
        from metfish.analysis.analysis_draft import DataPaths
        
        assert isinstance(DataPaths.BASE, Path)
        assert isinstance(DataPaths.REF, Path)
        assert isinstance(DataPaths.ENSEMBLE, Path)
    
    def test_paths_have_correct_structure(self):
        """Test path structure."""
        from metfish.analysis.analysis_draft import DataPaths
        
        assert "100125_Nature_Com_data" in str(DataPaths.BASE)
        assert DataPaths.REF.parent.name == "Apo_holo_data"


class TestApoHoloPairAnalyzer:
    """Test ApoHoloPairAnalyzer class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default paths."""
        from metfish.analysis.analysis_draft import ApoHoloPairAnalyzer, DataPaths
        
        with patch('pathlib.Path.exists', return_value=False):
            analyzer = ApoHoloPairAnalyzer()
            assert analyzer.ref_path == DataPaths.REF
            assert 'AlphaSAXS' in analyzer.model_paths
            assert 'OpenFold' in analyzer.model_paths
    
    def test_add_openfold_accuracy_categories(self):
        """Test OpenFold accuracy categorization."""
        from metfish.analysis.analysis_draft import ApoHoloPairAnalyzer
        
        with patch('metfish.analysis.analysis_draft.DataPaths') as mock_paths:
            mock_paths.APO_HOLO_CSV.exists.return_value = False
            analyzer = ApoHoloPairAnalyzer()
        
        df = pd.DataFrame({
            'pdb_id': ['A', 'B', 'C'],
            'type': ['OpenFold', 'OpenFold', 'OpenFold'],
            'rmsd': [0.5, 3.0, 7.0],
        })
        
        result = analyzer.add_openfold_accuracy(df)
        
        assert result.loc[0, 'OpenFold Accuracy'] == 'High'
        assert result.loc[1, 'OpenFold Accuracy'] == 'Medium'
        assert result.loc[2, 'OpenFold Accuracy'] == 'Low'


class TestEnsembleAnalyzer:
    """Test EnsembleAnalyzer class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default paths."""
        from metfish.analysis.analysis_draft import EnsembleAnalyzer, DataPaths
        
        analyzer = EnsembleAnalyzer()
        assert analyzer.ref_path == DataPaths.REF
        assert analyzer.ensemble_path == DataPaths.ENSEMBLE
    
    def test_extract_ensemble_file_not_found(self):
        """Test extract_ensemble raises error when file not found."""
        from metfish.analysis.analysis_draft import EnsembleAnalyzer
        
        analyzer = EnsembleAnalyzer(ensemble_path=Path("/nonexistent"))
        
        with pytest.raises(FileNotFoundError):
            analyzer.extract_ensemble("test_pdb")


class TestContactMapAnalyzer:
    """Test ContactMapAnalyzer class."""
    
    def test_flatten_contact_map(self):
        """Test contact map flattening."""
        from metfish.analysis.analysis_draft import ContactMapAnalyzer
        
        contact_map = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        result = ContactMapAnalyzer.flatten(contact_map)
        expected = np.array([1, 0, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity for identical maps."""
        from metfish.analysis.analysis_draft import ContactMapAnalyzer
        
        cm = np.array([True, False, True, True])
        result = ContactMapAnalyzer.jaccard_similarity(cm, cm)
        assert result == 1.0


class TestDataManager:
    """Test DataManager class."""
    
    def test_init_creates_cache_dir(self, tmp_path):
        """Test that initialization creates cache directory."""
        from metfish.analysis.analysis_draft import DataManager
        
        cache_dir = tmp_path / "test_cache"
        dm = DataManager(cache_dir=cache_dir)
        assert cache_dir.exists()
    
    def test_save_and_load_pkl(self, tmp_path):
        """Test pickle save and load."""
        from metfish.analysis.analysis_draft import DataManager
        
        dm = DataManager(cache_dir=tmp_path)
        test_data = {'key': 'value', 'numbers': [1, 2, 3]}
        
        dm.save_pkl(test_data, 'test.pkl')
        loaded = dm.load_pkl('test.pkl')
        
        assert loaded == test_data
    
    def test_save_and_load_csv(self, tmp_path):
        """Test CSV save and load."""
        from metfish.analysis.analysis_draft import DataManager
        
        dm = DataManager(cache_dir=tmp_path)
        test_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        
        dm.save_csv(test_df, 'test.csv')
        loaded = dm.load_csv('test.csv')
        
        pd.testing.assert_frame_equal(loaded, test_df)
    
    def test_load_nonexistent_returns_none(self, tmp_path):
        """Test loading nonexistent file returns None."""
        from metfish.analysis.analysis_draft import DataManager
        
        dm = DataManager(cache_dir=tmp_path)
        assert dm.load_pkl('nonexistent.pkl') is None
        assert dm.load_csv('nonexistent.csv') is None
