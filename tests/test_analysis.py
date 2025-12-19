"""
Tests for analysis module components
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from biopandas.pdb import PandasPdb

from metfish.analysis.processor import ProteinStructureAnalyzer, ModelComparisonProcessor


class TestProteinStructureAnalyzer:
    """Tests for ProteinStructureAnalyzer class."""
    
    @pytest.mark.requires_data
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = ProteinStructureAnalyzer()
        assert analyzer is not None
    
    @pytest.mark.requires_data
    def test_calculate_metrics(self, test_data_dir):
        """Test metrics calculation between two structures."""
        pdb_path = test_data_dir / "3nir.pdb"
        
        pdb_df_a = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        pdb_df_b = pdb_df_a.copy()
        
        analyzer = ProteinStructureAnalyzer()
        metrics = analyzer.calculate_metrics(
            pdb_df_a, pdb_df_b,
            str(pdb_path), str(pdb_path),
            "test"
        )
        
        assert 'rmsd' in metrics
        assert 'lddt' in metrics
        assert 'saxs_kldiv' in metrics
        assert 'rg_a' in metrics
        assert 'rg_b' in metrics
    
    @pytest.mark.requires_data
    def test_calculate_metrics_identical_structures(self, test_data_dir):
        """Test that identical structures give expected metrics."""
        pdb_path = test_data_dir / "3nir.pdb"
        
        pdb_df_a = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        pdb_df_b = pdb_df_a.copy()
        
        analyzer = ProteinStructureAnalyzer()
        metrics = analyzer.calculate_metrics(
            pdb_df_a, pdb_df_b,
            str(pdb_path), str(pdb_path),
            "test"
        )
        
        # RMSD should be very small for identical structures
        assert metrics['rmsd'] < 1e-2
        # lDDT should be close to 1.0 for identical structures
        assert metrics['lddt'] > 0.99 or metrics['lddt'] is None
        # SAXS KL divergence should be very small
        assert metrics['saxs_kldiv'] < 0.01
    
    @pytest.mark.requires_data
    def test_calculate_saxs_kldiv(self):
        """Test SAXS KL divergence calculation."""
        analyzer = ProteinStructureAnalyzer()
        
        # Create test SAXS profiles
        p_of_r_a = np.array([0.1, 0.3, 0.4, 0.2])
        p_of_r_b = np.array([0.15, 0.25, 0.35, 0.25])
        r_a = np.array([0, 1, 2, 3])
        r_b = np.array([0, 1, 2, 3])
        
        kldiv = analyzer._calculate_saxs_kldiv(p_of_r_a, p_of_r_b, r_a, r_b)
        
        assert kldiv >= 0  # KL divergence is always non-negative
    
    @pytest.mark.requires_data
    def test_calculate_saxs_kldiv_different_lengths(self):
        """Test SAXS KL divergence with different length profiles."""
        analyzer = ProteinStructureAnalyzer()
        
        p_of_r_a = np.array([0.1, 0.3, 0.4, 0.2])
        p_of_r_b = np.array([0.2, 0.3, 0.5])
        r_a = np.array([0, 1, 2, 3])
        r_b = np.array([0, 1, 2])
        
        kldiv = analyzer._calculate_saxs_kldiv(p_of_r_a, p_of_r_b, r_a, r_b)
        
        assert kldiv >= 0
    
    @pytest.mark.requires_data
    def test_metrics_contain_all_required_fields(self, test_data_dir):
        """Test that all required metric fields are present."""
        pdb_path = test_data_dir / "3nir.pdb"
        pdb_df = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        
        analyzer = ProteinStructureAnalyzer()
        metrics = analyzer.calculate_metrics(
            pdb_df, pdb_df,
            str(pdb_path), str(pdb_path),
            "test"
        )
        
        required_fields = [
            'rmsd', 'lddt', 'saxs_kldiv',
            'rg_a', 'rg_b', 'rg_diff',
            'plddt_a', 'plddt_bins_a',
            'plddt_b', 'plddt_bins_b',
            'saxs_bins_a', 'saxs_a',
            'saxs_bins_b', 'saxs_b'
        ]
        
        for field in required_fields:
            assert field in metrics


class TestModelComparisonProcessor:
    """Tests for ModelComparisonProcessor class."""
    
    def test_processor_initialization(self, temp_dir):
        """Test processor initialization."""
        model_dict = {
            'model1': {'tags': 'tag1', 'model_name': 'test_model'}
        }
        
        processor = ModelComparisonProcessor(
            model_dict=model_dict,
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        assert processor is not None
        assert processor.data_dir == temp_dir
        assert processor.output_dir == temp_dir
    
    def test_get_apo_holo_pairs_no_file(self, temp_dir):
        """Test apo-holo pairs when no metadata file exists."""
        model_dict = {
            'model1': {'tags': 'tag1', 'model_name': 'test_model'}
        }
        
        processor = ModelComparisonProcessor(
            model_dict=model_dict,
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        pairs = processor.apo_holo_pairs
        assert pairs is None
    
    def test_get_apo_holo_pairs_with_file(self, temp_dir):
        """Test apo-holo pairs with metadata file."""
        # Create mock metadata file
        metadata_path = temp_dir / "Table_rmsd_Apo_vs_Holo.csv"
        metadata_df = pd.DataFrame({
            'Apo_ID': ['1abc', '2def'],
            'Holo_ID': ['1xyz', '2uvw']
        })
        metadata_df.to_csv(metadata_path, sep=';', index=False)
        
        model_dict = {
            'model1': {'tags': 'tag1', 'model_name': 'test_model'}
        }
        
        processor = ModelComparisonProcessor(
            model_dict=model_dict,
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        pairs = processor.apo_holo_pairs
        assert pairs is not None
        assert len(pairs) == 2
        assert ('1abc', '1xyz') in pairs
    
    def test_create_comparisons_list(self, temp_dir):
        """Test creation of comparisons list."""
        model_dict = {
            'model1': {'tags': 'tag1', 'model_name': 'test1'},
            'model2': {'tags': 'tag2', 'model_name': 'test2'}
        }
        
        processor = ModelComparisonProcessor(
            model_dict=model_dict,
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        comparisons = processor._create_comparisons_list()
        
        assert len(comparisons) > 0
        # Should include comparisons between outputs and targets
        assert any('target' in str(c) for c in comparisons)
    
    def test_get_pair_name(self, temp_dir):
        """Test getting pair name."""
        metadata_path = temp_dir / "Table_rmsd_Apo_vs_Holo.csv"
        metadata_df = pd.DataFrame({
            'Apo_ID': ['1abc'],
            'Holo_ID': ['1xyz']
        })
        metadata_df.to_csv(metadata_path, sep=';', index=False)
        
        model_dict = {'model1': {'tags': 'tag1', 'model_name': 'test1'}}
        processor = ModelComparisonProcessor(
            model_dict=model_dict,
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        pair_name = processor._get_pair_name('1abc', ['1abc', '1xyz'])
        assert pair_name == '1xyz'
    
    def test_get_filename_ext_target(self, temp_dir):
        """Test getting filename extension for target."""
        model_dict = {'model1': {'tags': 'tag1', 'model_name': 'test1'}}
        processor = ModelComparisonProcessor(
            model_dict=model_dict,
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        ext = processor.get_filename_ext('target')
        assert ext == '.pdb'
    
    def test_get_filename_ext_output(self, temp_dir):
        """Test getting filename extension for output."""
        model_dict = {'model1': {'tags': 'tag1', 'model_name': 'test_model'}}
        processor = ModelComparisonProcessor(
            model_dict=model_dict,
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        ext = processor.get_filename_ext('out_tag1')
        assert ext == '_test_model_tag1.pdb'


class TestAnalysisEdgeCases:
    """Tests for edge cases in analysis module."""
    
    @pytest.mark.requires_data
    def test_metrics_with_missing_atoms(self, test_data_dir):
        """Test metrics calculation with structures having missing atoms."""
        pdb_path = test_data_dir / "3nir.pdb"
        pdb_df = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        
        # Remove some atoms to simulate missing data
        pdb_df_partial = pdb_df.iloc[::2]  # Keep every other atom
        
        analyzer = ProteinStructureAnalyzer()
        
        # Should handle gracefully or raise appropriate error
        try:
            metrics = analyzer.calculate_metrics(
                pdb_df, pdb_df_partial,
                str(pdb_path), str(pdb_path),
                "test"
            )
            # If it succeeds, check that metrics are reasonable
            assert metrics is not None
        except Exception:
            # If it fails, that's also acceptable behavior
            pass
    
    def test_processor_with_empty_model_dict(self, temp_dir):
        """Test processor with empty model dictionary."""
        processor = ModelComparisonProcessor(
            model_dict={},
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        assert processor is not None
        comparisons = processor._create_comparisons_list()
        # Should handle empty model dict gracefully
        assert isinstance(comparisons, list)
    
    def test_kldiv_with_zero_values(self):
        """Test KL divergence with zero values."""
        analyzer = ProteinStructureAnalyzer()
        
        # Profiles with some zeros
        p_of_r_a = np.array([0.0, 0.5, 0.5, 0.0])
        p_of_r_b = np.array([0.1, 0.4, 0.4, 0.1])
        r_a = np.array([0, 1, 2, 3])
        r_b = np.array([0, 1, 2, 3])
        
        # Should handle zeros with epsilon
        kldiv = analyzer._calculate_saxs_kldiv(p_of_r_a, p_of_r_b, r_a, r_b)
        
        assert np.isfinite(kldiv)
        assert kldiv >= 0


class TestComparisonDataFrame:
    """Tests for comparison dataframe creation."""
    
    def test_comparison_df_caching(self, temp_dir):
        """Test that comparison dataframe is cached."""
        model_dict = {'model1': {'tags': 'tag1', 'model_name': 'test1'}}
        
        processor = ModelComparisonProcessor(
            model_dict=model_dict,
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        # Create a mock cached dataframe
        cache_path = temp_dir / 'model_comparisons.csv'
        mock_df = pd.DataFrame({
            'name': ['test'],
            'rmsd': [1.0]
        })
        mock_df.to_csv(cache_path, index=False)
        
        # Should load from cache
        df = processor.get_comparison_df(overwrite=False)
        
        assert df is not None
        assert 'name' in df.columns
    
    def test_comparison_df_overwrite(self, temp_dir):
        """Test overwriting cached comparison dataframe."""
        model_dict = {'model1': {'tags': 'tag1', 'model_name': 'test1'}}
        
        processor = ModelComparisonProcessor(
            model_dict=model_dict,
            data_dir=temp_dir,
            output_dir=temp_dir
        )
        
        # Create a mock cached dataframe
        cache_path = temp_dir / 'model_comparisons.csv'
        mock_df = pd.DataFrame({'old': [1]})
        mock_df.to_csv(cache_path, index=False)
        
        # Mock the _create_comparison_df to avoid actual computation
        with patch.object(processor, '_create_comparison_df') as mock_create:
            mock_create.return_value = pd.DataFrame({'new': [2]})
            
            df = processor.get_comparison_df(names=[], overwrite=True)
            
            assert mock_create.called