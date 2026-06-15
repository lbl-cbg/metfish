"""
Integration tests for the metfish package
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_pdb_to_saxs_workflow(self, test_data_dir, temp_dir):
        """Test complete workflow from PDB to SAXS profile."""
        from metfish.utils import get_Pr
        
        pdb_path = test_data_dir / "3nir.pdb"
        
        # Calculate SAXS profile
        r, p = get_Pr(str(pdb_path), step=0.5)
        
        # Verify output
        assert len(r) == len(p)
        assert len(r) > 0
        assert np.all(r >= 0)
        assert np.all(p >= 0)
        
        # Save to file
        output_path = temp_dir / "saxs_profile.csv"
        df = pd.DataFrame({'r': r, 'P(r)': p})
        df.to_csv(output_path, index=False)
        
        assert output_path.exists()
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_structure_comparison_workflow(self, test_data_dir):
        """Test structure comparison workflow."""
        from metfish.utils import get_rmsd, get_lddt
        from biopandas.pdb import PandasPdb
        
        pdb_path = test_data_dir / "3nir.pdb"
        pdb_df_a = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        pdb_df_b = pdb_df_a.copy()
        
        # Calculate metrics
        rmsd = get_rmsd(pdb_df_a, pdb_df_b)
        lddt = get_lddt(pdb_df_a, pdb_df_b)
        
        # Verify results
        assert rmsd is not None
        assert rmsd < 1e-2  # Identical structures
        
        if lddt is not None:
            assert 0 <= lddt <= 1
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_sequence_extraction_workflow(self, test_data_dir, temp_dir):
        """Test sequence extraction workflow."""
        from metfish.utils import extract_seq
        
        pdb_path = test_data_dir / "3nir.pdb"
        output_path = temp_dir / "sequence.fasta"
        
        # Extract sequence
        extract_seq(str(pdb_path), str(output_path))
        
        # Verify output
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) >= 2
        assert lines[0].startswith('>')
        assert len(lines[1].strip()) > 0
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_conformer_generation_workflow(self, test_data_dir, temp_dir):
        """Test conformer generation workflow."""
        from metfish.utils import sample_conformers, write_conformers
        
        pdb_path = test_data_dir / "3nir.pdb"
        
        # Generate conformers
        protein = sample_conformers(
            str(pdb_path),
            n_modes=1,
            n_confs=2,
            rmsd=1.0
        )
        
        # Write conformers
        filenames = write_conformers(temp_dir, "test", protein)
        
        # Verify output
        assert len(filenames) > 0
        assert all(Path(f).exists() for f in filenames)
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_cli_to_analysis_workflow(self, test_data_dir, temp_dir):
        """Test workflow from CLI to analysis."""
        from metfish.utils import get_Pr
        from metfish.analysis.processor import ProteinStructureAnalyzer
        from biopandas.pdb import PandasPdb
        
        pdb_path = test_data_dir / "3nir.pdb"
        
        # Step 1: Calculate SAXS
        r, p = get_Pr(str(pdb_path))
        assert len(r) > 0
        
        # Step 2: Load structure
        pdb_df = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        
        # Step 3: Analyze structure
        analyzer = ProteinStructureAnalyzer()
        metrics = analyzer.calculate_metrics(
            pdb_df, pdb_df,
            str(pdb_path), str(pdb_path),
            "test"
        )
        
        # Verify complete workflow
        assert 'rmsd' in metrics
        assert 'saxs_kldiv' in metrics
        assert len(metrics['saxs_a']) > 0


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    @pytest.mark.requires_torch
    @pytest.mark.requires_openfold
    def test_pdb_feature_extraction(self, test_data_dir):
        """Test PDB feature extraction pipeline."""
        from metfish.msa_model.data.data_pipeline import DataPipeline
        
        pdb_path = test_data_dir / "3nir.pdb"
        
        # Create pipeline
        pipeline = DataPipeline(template_featurizer=None)
        
        # Extract features
        features = pipeline.process_pdb_feats(str(pdb_path), is_distillation=False)
        
        # Verify features
        assert 'aatype' in features
        assert 'all_atom_positions' in features
        assert 'all_atom_mask' in features
        
        # Verify shapes are consistent
        num_res = features['aatype'].shape[0]
        assert features['all_atom_positions'].shape[0] == num_res
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    @pytest.mark.requires_torch
    def test_saxs_computation_pipeline(self, test_data_dir):
        """Test SAXS computation through full pipeline."""
        from metfish.msa_model.utils.loss import compute_saxs
        from metfish.msa_model.data.data_pipeline import DataPipeline
        from metfish.utils import get_Pr
        import torch
        
        pdb_path = test_data_dir / "3nir.pdb"
        
        # Get reference SAXS from get_Pr
        r_ref, p_ref = get_Pr(str(pdb_path), step=0.5, dmax=128)
        
        # Get SAXS through data pipeline
        pipeline = DataPipeline(template_featurizer=None)
        features = pipeline.process_pdb_feats(str(pdb_path), is_distillation=False)
        
        all_atom_pos = torch.tensor(features["all_atom_positions"], dtype=torch.float32)
        all_atom_mask = torch.tensor(features["all_atom_mask"], dtype=torch.float32)
        
        p_computed = compute_saxs(all_atom_pos, all_atom_mask, step=0.5, dmax=128)
        
        # Compare results
        p_ref_tensor = torch.tensor(p_ref, dtype=torch.float32)
        assert torch.allclose(p_computed, p_ref_tensor, atol=0.05)


class TestModelConfigIntegration:
    """Integration tests for model configurations."""
    
    @pytest.mark.integration
    def test_all_msa_model_configs(self):
        """Test that all MSA model configurations are valid."""
        from metfish.msa_model.config import model_config
        
        model_names = [
            "generating", "finetuning", "finetuning_ptm",
            "finetuning_no_templ", "finetuning_no_templ_ptm",
            "model_1", "model_2", "model_3", "model_4", "model_5",
            "model_1_ptm", "model_2_ptm", "model_3_ptm", "model_4_ptm", "model_5_ptm"
        ]
        
        for name in model_names:
            if "finetuning" in name or "generating" in name:
                config = model_config(name, train=True)
            else:
                config = model_config(name, train=False)
            
            assert config is not None
            assert hasattr(config, 'data')
            assert hasattr(config, 'model')
            assert hasattr(config, 'loss')
    
    @pytest.mark.integration
    def test_all_refinement_model_configs(self):
        """Test that all refinement model configurations are valid."""
        from metfish.refinement_model.config import model_config
        
        model_names = [
            "initial_training", "finetuning", "finetuning_ptm",
            "finetuning_no_templ", "finetuning_no_templ_ptm",
            "model_1", "model_2", "model_3", "model_4", "model_5",
            "model_1_ptm", "model_2_ptm", "model_3_ptm", "model_4_ptm", "model_5_ptm"
        ]
        
        for name in model_names:
            if "finetuning" in name or "initial_training" in name:
                config = model_config(name, train=True)
            else:
                config = model_config(name, train=False)
            
            assert config is not None
            assert hasattr(config, 'data')
            assert hasattr(config, 'model')
            assert hasattr(config, 'loss')


class TestCrossModuleIntegration:
    """Integration tests across multiple modules."""
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_utils_and_analysis_integration(self, test_data_dir):
        """Test integration between utils and analysis modules."""
        from metfish.utils import get_Pr, get_rmsd
        from metfish.analysis.processor import ProteinStructureAnalyzer
        from biopandas.pdb import PandasPdb
        
        pdb_path = test_data_dir / "3nir.pdb"
        
        # Use utils
        r, p = get_Pr(str(pdb_path))
        pdb_df = PandasPdb().read_pdb(str(pdb_path)).df['ATOM']
        rmsd = get_rmsd(pdb_df, pdb_df)
        
        # Use analysis
        analyzer = ProteinStructureAnalyzer()
        kldiv = analyzer._calculate_saxs_kldiv(p, p, r, r)
        
        assert rmsd < 1e-2
        assert kldiv < 0.01  # Identical profiles
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    @pytest.mark.requires_torch
    def test_config_and_loss_integration(self, test_data_dir):
        """Test integration between config and loss modules."""
        from metfish.msa_model.config import model_config
        from metfish.msa_model.utils.loss import compute_saxs
        from metfish.msa_model.data.data_pipeline import DataPipeline
        import torch
        
        # Get config
        config = model_config("model_1", train=False)
        
        # Use config parameters
        dmax = config.loss.saxs_loss.dmax
        step = config.loss.saxs_loss.step
        
        # Process data
        pdb_path = test_data_dir / "3nir.pdb"
        pipeline = DataPipeline(template_featurizer=None)
        features = pipeline.process_pdb_feats(str(pdb_path), is_distillation=False)
        
        all_atom_pos = torch.tensor(features["all_atom_positions"], dtype=torch.float32)
        all_atom_mask = torch.tensor(features["all_atom_mask"], dtype=torch.float32)
        
        # Compute SAXS using config parameters
        saxs = compute_saxs(all_atom_pos, all_atom_mask, step=step, dmax=dmax)
        
        assert saxs is not None
        assert len(saxs) > 0


class TestRobustness:
    """Robustness and stress tests."""
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_multiple_structures_processing(self, test_data_dir):
        """Test processing multiple structures in sequence."""
        from metfish.utils import get_Pr
        
        pdb_files = [
            test_data_dir / "3nir.pdb",
            test_data_dir / "3DB7_A.pdb",
            test_data_dir / "3M8J_A.pdb"
        ]
        
        results = []
        for pdb_file in pdb_files:
            if pdb_file.exists():
                r, p = get_Pr(str(pdb_file))
                results.append((r, p))
        
        # All should succeed
        assert len(results) > 0
        
        # All should have valid outputs
        for r, p in results:
            assert len(r) > 0
            assert len(p) > 0
            assert np.all(r >= 0)
            assert np.all(p >= 0)
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_error_recovery(self, test_data_dir, temp_dir):
        """Test error recovery in workflows."""
        from metfish.utils import extract_seq
        
        # Test with invalid file
        with pytest.raises(Exception):
            extract_seq("nonexistent.pdb", str(temp_dir / "out.fasta"))
        
        # Test with valid file should still work
        pdb_path = test_data_dir / "3nir.pdb"
        output_path = temp_dir / "sequence.fasta"
        extract_seq(str(pdb_path), str(output_path))
        
        assert output_path.exists()


class TestPerformance:
    """Performance-related integration tests."""
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_large_structure_handling(self, test_data_dir):
        """Test handling of larger structures."""
        from metfish.utils import get_Pr
        
        # Use the largest available test structure
        pdb_path = test_data_dir / "3nir.pdb"
        
        # Should complete without timeout
        r, p = get_Pr(str(pdb_path), step=0.5)
        
        assert len(r) > 0
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_repeated_operations(self, test_data_dir):
        """Test repeated operations for consistency."""
        from metfish.utils import get_Pr
        
        pdb_path = test_data_dir / "3nir.pdb"
        
        # Run multiple times
        results = []
        for _ in range(3):
            r, p = get_Pr(str(pdb_path), step=0.5)
            results.append((r, p))
        
        # Results should be identical
        r0, p0 = results[0]
        for r, p in results[1:]:
            np.testing.assert_array_equal(r, r0)
            np.testing.assert_array_almost_equal(p, p0)