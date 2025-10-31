import argparse
import pandas as pd

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
    skip_inference: bool = False,
    models: Tuple[str] = ('AlphaFold', 'SFold_NMR', 'SFold_NMA'),
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
    if not skip_inference:
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
    
    names = pd.read_csv(data_dir / 'input_all.csv')['name'].tolist()
    comparison_df = processor.get_comparison_df(names=names)

    # Visualize results
    color_scheme = {"NMR": "#264882", "NMA": "#b13c6c", "AF": "#56994A", "Target": "#5c5c5c", }
    
    viz = ProteinVisualization(comparison_df, color_scheme)
    viz.plot_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run protein structure analysis')
    parser.add_argument('--data-dir', type=str, default=Path("/global/cfs/cdirs/m4704/100125_Nature_Com_data/Apo_holo_data"), help='Data directory')
    parser.add_argument('--ckpt-dir', type=str, default=Path("/global/cfs/cdirs/m4704/100125_Nature_Com_data/single_conformation"), help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default=Path("/global/cfs/cdirs/m4704/100125_Nature_Com_data/results"), help='Output directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing files')
    parser.add_argument('--skip-inference', action='store_true', default=True, help='Skip inference step')
    parser.add_argument('--models', type=str, nargs='+', default=['SFold_NMR', 'SFold_NMA'], help='Models to run')

    args = parser.parse_args()
    
    main(
        data_dir=Path(args.data_dir),
        ckpt_dir=Path(args.ckpt_dir),
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        skip_inference=args.skip_inference,
        models=args.models
    )