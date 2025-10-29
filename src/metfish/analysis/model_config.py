from typing import Dict

class ModelConfig:
    """Configuration for model inference and comparison."""
    
    def __init__(self, model_configs: Dict[str, Dict]):
        """
        Args:
            model_configs: Dictionary mapping model names to their configuration parameters
        """
        self.model_configs = model_configs
        
    def get_filename_ext(self, tag: str) -> str:
        """Get filename extension based on model tag."""
        tags_to_keys = {v['tags']: k for k, v in self.model_configs.items()}
        
        for k, v in tags_to_keys.items():
            if f"out_{k}" == tag:
                return f"_{self.model_configs[v]['model_name']}_{k}_unrelaxed.pdb"
            elif k in tag:
                return f"_{self.model_configs[v]['model_name']}_{k}_unrelaxed.pdb"
            elif 'target' in tag:
                return self.model_configs[v]['pdb_ext']
        
        raise ValueError(f"Unknown tag: {tag}")