"""
Tests for MSA model components
"""
import pytest
import torch
import numpy as np

from metfish.msa_model.config import model_config


class TestMSAModelConfig:
    """Tests for MSA model configuration."""
    
    def test_model_config_initial_training(self):
        """Test initial training configuration."""
        config = model_config("generating", train=True)
        
        assert config is not None
        assert config.globals.blocks_per_ckpt == 1
        assert config.globals.chunk_size is None
    
    def test_model_config_finetuning(self):
        """Test finetuning configuration."""
        config = model_config("finetuning", train=True)
        
        assert config.data.train.crop_size == 384
        assert config.data.train.max_extra_msa == 5120
        assert config.loss.violation.weight == 1.0
    
    def test_model_config_inference_model_1(self):
        """Test model 1 inference configuration."""
        config = model_config("model_1", train=False)
        
        assert config.data.predict.max_extra_msa == 5120
        assert config.model.template.enabled is True
    
    def test_model_config_inference_model_3(self):
        """Test model 3 (no templates) configuration."""
        config = model_config("model_3", train=False)
        
        assert config.model.template.enabled is False
    
    def test_model_config_ptm_enabled(self):
        """Test PTM-enabled configuration."""
        config = model_config("model_1_ptm", train=False)
        
        assert config.model.heads.tm.enabled is True
        assert config.loss.tm.weight == 0.1
    
    def test_model_config_long_sequence(self):
        """Test long sequence inference configuration."""
        config = model_config("model_1", train=False, long_sequence_inference=True)
        
        assert config.globals.offload_inference is True
        assert config.globals.use_lma is True
        assert config.globals.use_flash is False
    
    def test_model_config_low_precision(self):
        """Test low precision configuration."""
        config = model_config("model_1", train=False, low_prec=True)
        
        assert config.globals.eps == 1e-4
    
    def test_model_config_deterministic(self):
        """Test deterministic configuration."""
        config = model_config("model_1", train=False, deterministic=True)
        
        assert config.data.eval.masked_msa_replace_fraction == 0.0
        assert config.model.global_config.deterministic is True
    
    def test_model_config_saxs_loss(self):
        """Test SAXS loss configuration."""
        config = model_config("model_1", train=True, use_l1_loss=True)
        
        assert config.loss.saxs_loss.use_l1 is True
        assert config.loss.saxs_loss.weight == 2
    
    def test_model_config_saxs_loss_only(self):
        """Test SAXS loss only configuration."""
        config = model_config("model_1", train=True, use_saxs_loss_only=True)
        
        assert config.loss.saxs_loss_only is True
    
    def test_model_config_custom_saxs_padding(self):
        """Test custom SAXS padding configuration."""
        saxs_padding = 512
        config = model_config("model_1", train=True, saxs_padding=saxs_padding)
        
        assert config.data.common.feat.saxs == [saxs_padding]
        assert config.loss.saxs_loss.dmax == int(saxs_padding * config.loss.saxs_loss.step)
    
    def test_model_config_invalid_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Invalid model name"):
            model_config("invalid_model")
    
    def test_model_config_mutually_exclusive_flags(self):
        """Test that mutually exclusive flags raise error."""
        # This should raise an error if both use_lma and use_flash are True
        # Note: This may not be directly testable without modifying config after creation
        pass


@pytest.mark.requires_torch
class TestMSALoss:
    """Tests for MSA model loss functions."""
    
    def test_differentiable_histogram(self):
        """Test differentiable histogram computation."""
        from metfish.msa_model.utils.loss import differentiable_histogram
        
        values = torch.tensor([0.5, 1.5, 2.5])
        weights = torch.tensor([1.0, 2.0, 1.0])
        bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0])
        
        hist = differentiable_histogram(values, weights, bin_edges)
        
        expected_hist = torch.tensor([1.0, 2.0, 1.0])
        assert torch.allclose(hist, expected_hist, atol=1e-1)
    
    def test_differentiable_histogram_gradient(self):
        """Test that differentiable histogram maintains gradients."""
        from metfish.msa_model.utils.loss import differentiable_histogram
        
        values = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
        weights = torch.tensor([1.0, 2.0, 1.0])
        bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0])
        
        hist = differentiable_histogram(values, weights, bin_edges)
        loss = hist.sum()
        loss.backward()
        
        assert values.grad is not None
    
    @pytest.mark.requires_data
    def test_compute_saxs(self, test_data_dir):
        """Test SAXS computation from atomic positions."""
        from metfish.msa_model.utils.loss import compute_saxs
        from metfish.msa_model.data.data_pipeline import DataPipeline
        
        pdb_path = test_data_dir / "3DB7_A.pdb"
        
        data_pipeline = DataPipeline(template_featurizer=None)
        pdb_features = data_pipeline.process_pdb_feats(str(pdb_path), is_distillation=False)
        
        all_atom_pos = torch.tensor(pdb_features["all_atom_positions"], dtype=torch.float32)
        all_atom_mask = torch.tensor(pdb_features["all_atom_mask"], dtype=torch.float32)
        
        hist = compute_saxs(all_atom_pos, all_atom_mask, step=0.5, dmax=256)
        
        assert hist is not None
        assert len(hist) > 0
        assert torch.all(hist >= 0)
    
    @pytest.mark.requires_data
    def test_compute_saxs_vs_get_pr(self, test_data_dir):
        """Test that compute_saxs matches get_Pr."""
        from metfish.msa_model.utils.loss import compute_saxs
        from metfish.msa_model.data.data_pipeline import DataPipeline
        from metfish.utils import get_Pr
        from Bio.PDB import PDBParser
        
        step = 0.5
        dmax = 256
        
        pdb_path = test_data_dir / "3DB7_A.pdb"
        structure = PDBParser().get_structure("", str(pdb_path))
        bins, expected_hist = get_Pr(structure, step=step, dmax=dmax)
        expected_hist = torch.tensor(expected_hist, dtype=torch.float32)
        
        data_pipeline = DataPipeline(template_featurizer=None)
        pdb_features = data_pipeline.process_pdb_feats(str(pdb_path), is_distillation=False)
        all_atom_pos = torch.tensor(pdb_features["all_atom_positions"], dtype=torch.float32)
        all_atom_mask = torch.tensor(pdb_features["all_atom_mask"], dtype=torch.float32)
        hist = compute_saxs(all_atom_pos, all_atom_mask, step=step, dmax=dmax)
        
        assert torch.allclose(hist, expected_hist, atol=1e-2)
    
    def test_compute_saxs_requires_grad(self):
        """Test that compute_saxs maintains gradients."""
        from metfish.msa_model.utils.loss import compute_saxs
        
        all_atom_pos = torch.randn(1, 37, 3, requires_grad=True)
        all_atom_mask = torch.ones(1, 37)
        
        saxs_profile = compute_saxs(all_atom_pos, all_atom_mask)
        
        assert saxs_profile.requires_grad


@pytest.mark.requires_torch
class TestMSATensorUtils:
    """Tests for MSA tensor utilities."""
    
    def test_tensor_tree_map(self):
        """Test tensor tree mapping utility."""
        from metfish.msa_model.utils.tensor_utils import tensor_tree_map
        
        tree = {
            'a': torch.tensor([1, 2, 3]),
            'b': {
                'c': torch.tensor([4, 5, 6]),
                'd': torch.tensor([7, 8, 9])
            }
        }
        
        result = tensor_tree_map(lambda x: x * 2, tree)
        
        assert torch.equal(result['a'], torch.tensor([2, 4, 6]))
        assert torch.equal(result['b']['c'], torch.tensor([8, 10, 12]))


@pytest.mark.requires_torch
@pytest.mark.requires_openfold
class TestMSADataPipeline:
    """Tests for MSA data pipeline."""
    
    @pytest.mark.requires_data
    def test_data_pipeline_process_pdb(self, test_data_dir):
        """Test PDB feature processing."""
        from metfish.msa_model.data.data_pipeline import DataPipeline
        
        pdb_path = test_data_dir / "3nir.pdb"
        data_pipeline = DataPipeline(template_featurizer=None)
        
        features = data_pipeline.process_pdb_feats(str(pdb_path), is_distillation=False)
        
        assert 'aatype' in features
        assert 'all_atom_positions' in features
        assert 'all_atom_mask' in features
    
    @pytest.mark.requires_data
    def test_data_pipeline_feature_shapes(self, test_data_dir):
        """Test that feature shapes are correct."""
        from metfish.msa_model.data.data_pipeline import DataPipeline
        
        pdb_path = test_data_dir / "3nir.pdb"
        data_pipeline = DataPipeline(template_featurizer=None)
        
        features = data_pipeline.process_pdb_feats(str(pdb_path), is_distillation=False)
        
        num_res = features['aatype'].shape[0]
        assert features['all_atom_positions'].shape == (num_res, 37, 3)
        assert features['all_atom_mask'].shape == (num_res, 37)


@pytest.mark.requires_torch
class TestMSAValidationMetrics:
    """Tests for MSA validation metrics."""
    
    def test_validation_metrics_basic(self):
        """Test basic validation metrics computation."""
        # This would test specific validation metrics if available
        pass


class TestMSAEdgeCases:
    """Tests for MSA model edge cases."""
    
    def test_config_with_all_options(self):
        """Test configuration with all options enabled."""
        config = model_config(
            "model_1_ptm",
            train=True,
            low_prec=True,
            deterministic=True,
            use_l1_loss=True,
            use_saxs_loss_only=True,
            saxs_padding=512
        )
        
        assert config is not None
        assert config.loss.saxs_loss.use_l1 is True
        assert config.loss.saxs_loss_only is True