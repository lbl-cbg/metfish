from pathlib import Path

from metfish.analysis.model_config import ModelConfig
from metfish.analysis.processor import ModelComparisonProcessor, ProteinStructureAnalyzer
from metfish.analysis.visualizer import ProteinVisualization

def main():
 # Setup configuration
    model_dict = {
        'AlphaFold': {
            'model_name': 'AlphaFold',
            'tags': 'AF',
            'pdb_ext': '_atom_only.pdb'
        },
        'AFSAXS_NMRtrain': {
            'model_name': 'AFSAXS',
            'tags': 'NMRtrain',
            'pdb_ext': '_atom_only.pdb'
        }
    }
    
    config = ModelConfig(model_dict)
    analyzer = ProteinStructureAnalyzer(config)
    
    # Process comparisons
    processor = ModelComparisonProcessor(
        model_config=config,
        analyzer=analyzer,
        data_dir=Path("/path/to/data"),
        output_dir=Path("/path/to/output")
    )
    
    comparison_df = processor.create_comparison_df(
        pairs=[('apo1', 'holo1')],
        names=['apo1', 'holo1'],
        comparisons=[('out_AF', 'target'), ('out_NMRtrain', 'target')]
    )
    
    # Visualize results
    color_scheme = {"AF": "#2b2b2b", "SFold NMR": "#264882", "Target": "#5c5c5c"}
    label_dict = {"out_AF": "AF", "out_NMRtrain": "SFold NMR", "target": "Target"}
    
    viz = ProteinVisualization(color_scheme, label_dict)
    viz.plot_overall_metrics(comparison_df, ['AF', 'NMRtrain'], ['rmsd', 'lddt'])


if __name__ == "__main__":
   main()