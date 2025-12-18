"""
Tests for refinement model components
"""
import pytest
import torch
import numpy as np

from metfish.refinement_model.config import model_config


class TestRefinementModelConfig:
    """Tests for refinement model configuration."""
    
    def test_model_config_initial_training(self):
        """Test initial training configuration."""
        config = model_config("initial_training", train=True)
        
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
    
    def test_model_config_inference_model_5(self):
        """Test model 5 (no templates) configuration."""
        config = model_config("model_5", train=False)
        
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
    
    def test_model_config_saxs_settings(self):
        """Test SAXS-specific settings in refinement config."""
        config = model_config("model_1", train=True)
        
        # Check default SAXS settings
        assert config.data.common.feat.saxs == [512]
        assert config.loss.saxs_loss.dmax == 256
        assert config.loss.saxs_loss.step == 0.5
    
    def test_model_config_invalid_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Invalid model name"):
            model_config("invalid_model")
    
    def test_model_config_finetuning_no_templ(self):
        """Test finetuning without templates."""
        config = model_config("finetuning_no_templ", train=True)
        
        assert config.model.template.enabled is False
        assert config.data.train.crop_size == 384
    
    def test_model_config_finetuning_no_templ_ptm(self):
        """Test finetuning without templates with PTM."""
        config = model_config("finetuning_no_templ_ptm", train=True)
        
        assert config.model.template.enabled is False
        assert config.model.heads.tm.enabled is True
        assert config.loss.tm.weight == 0.1


@pytest.mark.requires_torch
class TestRandomModel:
    """Tests for random model generation."""
    
    def test_random_model_import(self):
        """Test that random model can be imported."""
        try:
            from metfish.refinement_model.random_model import RandomModel
            assert RandomModel is not None
        except ImportError:
            pytest.skip("RandomModel not available")
    
    @pytest.mark.slow
    def test_random_model_basic(self):
        """Test basic random model functionality."""
        try:
            from metfish.refinement_model.random_model import RandomModel
            
            # This is a placeholder - actual implementation may vary
            model = RandomModel()
            assert model is not None
        except (ImportError, AttributeError):
            pytest.skip("RandomModel not fully implemented")


@pytest.mark.requires_torch
@pytest.mark.requires_openfold
class TestModelWrapper:
    """Tests for model wrapper."""
    
    def test_model_wrapper_import(self):
        """Test that model wrapper can be imported."""
        try:
            from metfish.refinement_model.model_wrapper import ModelWrapper
            assert ModelWrapper is not None
        except ImportError:
            pytest.skip("ModelWrapper not available")


class TestConfigConsistency:
    """Tests for configuration consistency between models."""
    
    def test_config_feature_shapes(self):
        """Test that feature shapes are consistent."""
        config = model_config("model_1", train=False)
        
        feat = config.data.common.feat
        
        # Check that key features are defined
        assert 'aatype' in feat
        assert 'all_atom_positions' in feat
        assert 'all_atom_mask' in feat
        assert 'saxs' in feat
    
    def test_config_global_parameters(self):
        """Test global parameters."""
        config = model_config("model_1", train=False)
        
        assert hasattr(config.globals, 'c_z')
        assert hasattr(config.globals, 'c_m')
        assert hasattr(config.globals, 'eps')
    
    def test_config_loss_weights(self):
        """Test loss weight configuration."""
        config = model_config("model_1", train=True)
        
        assert hasattr(config.loss, 'fape')
        assert hasattr(config.loss, 'saxs_loss')
        assert hasattr(config.loss, 'distogram')
    
    def test_config_data_pipeline_settings(self):
        """Test data pipeline settings."""
        config = model_config("model_1", train=True)
        
        assert hasattr(config.data, 'train')
        assert hasattr(config.data, 'eval')
        assert hasattr(config.data, 'predict')
    
    def test_config_model_architecture(self):
        """Test model architecture settings."""
        config = model_config("model_1", train=False)
        
        assert hasattr(config.model, 'evoformer_stack')
        assert hasattr(config.model, 'structure_module')
        assert hasattr(config.model, 'heads')


class TestTrainingConfigurations:
    """Tests for training-specific configurations."""
    
    def test_train_mode_checkpointing(self):
        """Test checkpointing in training mode."""
        config = model_config("model_1", train=True)
        
        assert config.globals.blocks_per_ckpt == 1
    
    def test_train_mode_chunking(self):
        """Test chunking disabled in training mode."""
        config = model_config("model_1", train=True)
        
        assert config.globals.chunk_size is None
    
    def test_train_mode_memory_settings(self):
        """Test memory-efficient settings in training mode."""
        config = model_config("model_1", train=True)
        
        assert config.globals.use_lma is False
        assert config.globals.offload_inference is False
    
    def test_train_mode_template_settings(self):
        """Test template settings in training mode."""
        config = model_config("model_1", train=True)
        
        assert config.model.template.average_templates is False
        assert config.model.template.offload_templates is False


class TestInferenceConfigurations:
    """Tests for inference-specific configurations."""
    
    def test_inference_mode_settings(self):
        """Test inference mode settings."""
        config = model_config("model_1", train=False)
        
        # In inference mode, chunk_size should be set
        assert config.globals.chunk_size is not None or config.globals.use_lma
    
    def test_inference_long_sequence_settings(self):
        """Test long sequence inference settings."""
        config = model_config("model_1", train=False, long_sequence_inference=True)
        
        assert config.globals.offload_inference is True
        assert config.globals.use_lma is True
        assert config.model.template.offload_inference is True
    
    def test_inference_all_models(self):
        """Test that all model configurations can be created."""
        model_names = [
            "model_1", "model_2", "model_3", "model_4", "model_5",
            "model_1_ptm", "model_2_ptm", "model_3_ptm", "model_4_ptm", "model_5_ptm"
        ]
        
        for name in model_names:
            config = model_config(name, train=False)
            assert config is not None


class TestRefinementEdgeCases:
    """Tests for edge cases in refinement model."""
    
    def test_config_with_conflicting_settings(self):
        """Test that conflicting settings are handled properly."""
        # Long sequence inference should override train settings
        config = model_config("model_1", train=False, long_sequence_inference=True)
        
        assert config.globals.offload_inference is True
    
    def test_config_template_enabled_consistency(self):
        """Test template enabled/disabled consistency."""
        # Model 1 should have templates enabled
        config1 = model_config("model_1", train=False)
        assert config1.model.template.enabled is True
        
        # Model 5 should have templates disabled
        config5 = model_config("model_5", train=False)
        assert config5.model.template.enabled is False
    
    def test_config_ptm_settings_consistency(self):
        """Test PTM settings consistency."""
        # PTM model should have TM head enabled
        config_ptm = model_config("model_1_ptm", train=False)
        assert config_ptm.model.heads.tm.enabled is True
        
        # Non-PTM model should have TM head disabled
        config_no_ptm = model_config("model_1", train=False)
        assert config_no_ptm.model.heads.tm.enabled is False


class TestConfigParameters:
    """Tests for specific configuration parameters."""
    
    def test_saxs_loss_parameters(self):
        """Test SAXS loss parameters."""
        config = model_config("model_1", train=True)
        
        assert config.loss.saxs_loss.dmax == 256
        assert config.loss.saxs_loss.step == 0.5
        assert config.loss.saxs_loss.weight == 5.0
    
    def test_structure_module_parameters(self):
        """Test structure module parameters."""
        config = model_config("model_1", train=False)
        
        sm = config.model.structure_module
        assert sm.no_blocks == 8
        assert sm.no_heads_ipa == 12
    
    def test_evoformer_parameters(self):
        """Test evoformer stack parameters."""
        config = model_config("model_1", train=False)
        
        ef = config.model.evoformer_stack
        assert ef.no_blocks == 48
        assert ef.no_heads_msa == 8
        assert ef.no_heads_pair == 4