import argparse

from pathlib import Path
from typing import Tuple

from metfish.msa_model.predict import inference
from metfish.analysis.processor import ModelComparisonProcessor
from metfish.analysis.visualizer import ProteinVisualization

def main( 
    data_dir: Path,
    ckpt_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    models: Tuple[str] = ('AlphaFold', 'SFold_NMR', 'SFold_NMA')
):
     # Setup configuration
    model_dict = {
        'AlphaFold': {
            'model_name': 'AlphaFold',
            'tags': 'AF',
            'ckpt_path': ckpt_dir / 'af' / 'checkpoints' / 'params_model_1.npz'
        },
        'SFold_NMR': {
            'model_name': 'SFold',
            'tags': 'NMR',
            'ckpt_path': ckpt_dir / 'nmr' / 'checkpoints' / 'epoch=17-step=24282.ckpt',
        },
        'SFold_NMA': {
            'model_name': 'SFold',
            'tags': 'NMA',
            'ckpt_path': ckpt_dir / 'nma'/ 'checkpoints' / 'epoch=16-step=17595.ckpt'
        }
    }
    
    # TODO - replace old results archive with the new ones generated from that subset of 40 structures
    # Generate models
    for model_key, model_kwargs in model_dict.items():
        if model_key in models:
            print(f'Running inference for {model_key}...')
            inference(**model_kwargs, 
                    data_dir=data_dir,
                    output_dir=output_dir / model_kwargs['tags'],
                    test_csv_name=data_dir / 'input_all.csv',
                    overwrite=overwrite,
                    deterministic=True,
                    random_seed=1234,
                    pdb_ext='.pdb',
                    saxs_ext='_atom_only.csv',
                    save_output_dict=False)


    # Process comparisons
    processor = ModelComparisonProcessor(
        model_dict=model_dict,
        data_dir=data_dir,
        output_dir=output_dir
    )
    
    comparison_df = processor.create_comparison_df(
        pairs=[('apo1', 'holo1')],
        names=['apo1', 'holo1'],
        comparisons=[('out_AF', 'target'), ('out_NMRtrain', 'target')]
    )
    
    # Visualize results
    color_scheme = {"AF": "#56994A", "Target": "#5c5c5c", "SFold NMR": "#264882", "SFold NMA": "#b13c6c"}
    label_dict = {"out_AF": "AF", "out_NMR": "SFold NMR", "out_NMA": "SFold NMA", "target": "Target"}
    
    viz = ProteinVisualization(comparison_df, color_scheme, label_dict)
    viz.plot_overall_metrics(comparison_df, ['AF', 'NMRtrain', 'NMAtrain'], ['rmsd', 'lddt'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run protein structure analysis')
    parser.add_argument('--data-dir', type=str, default=Path("/global/cfs/cdirs/m4704/100125_Nature_Com_data/Apo_holo_data"), help='Data directory')
    parser.add_argument('--ckpt-dir', type=str, default=Path("/global/cfs/cdirs/m4704/100125_Nature_Com_data/single_conformation"), help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default=Path("/global/cfs/cdirs/m4704/100125_Nature_Com_data/results"), help='Output directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing files')
    parser.add_argument('--models', type=str, nargs='+', default=['AlphaFold', 'SFold_NMR', 'SFold_NMA'], help='Models to run')

    args = parser.parse_args()
    
    main(
        data_dir=Path(args.data_dir),
        ckpt_dir=Path(args.ckpt_dir),
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        models=args.models
    )