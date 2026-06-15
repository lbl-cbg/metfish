#!/usr/bin/env python3
"""
Generate All Figures
====================

Master script to regenerate all publication figures.

Usage:
    python generate_all_figures.py
    python generate_all_figures.py --figures figure2 figure3
    python generate_all_figures.py --output-dir custom_output/
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from metfish.analysis.plot_config import apply_plot_style
from metfish.analysis.run_analysis import (
    generate_figure2_rmsd,
    generate_figure2_rg,
    generate_figure2_saxs,
    generate_figure2_metrics_barplot,
    generate_figure3_pr,
    generate_figure3_rmsd,
    generate_figure3_recovery_barplot,
    generate_figure3_correlation,
    generate_apo_holo_pr_plots,
    generate_three_model_pr_plots,
    generate_figure4_barplot,
    generate_figure4_rg,
    generate_figure4_re,
    generate_figure4_diversity,
    OUTPUT_DIR,
)

# Set default output directory for publication figures (separate from analysis_output)
SCRIPT_OUTPUT_DIR = Path('publication_figures').resolve()


FIGURE_GROUPS = {
    'figure2': [
        ('Figure 2: RMSD Comparison', generate_figure2_rmsd),
        ('Figure 2: Rg Comparison', generate_figure2_rg),
        ('Figure 2: SAXS Comparison', generate_figure2_saxs),
        ('Figure 2: Metrics Barplot', generate_figure2_metrics_barplot),
    ],
    'figure3': [
        ('Figure 3: P(r) Divergence', generate_figure3_pr),
        ('Figure 3: Apo-Holo RMSD', generate_figure3_rmsd),
        ('Figure 3: Recovery Barplot', generate_figure3_recovery_barplot),
        ('Figure 3: Correlation', generate_figure3_correlation),
    ],
    'notebook_pr': [
        ('Apo-Holo P(r) Comparisons', generate_apo_holo_pr_plots),
        ('Three-Model P(r) Comparisons', generate_three_model_pr_plots),
    ],
    'figure4': [
        ('Figure 4: Ensemble Barplot', generate_figure4_barplot),
        ('Figure 4: Rg vs RMSD', generate_figure4_rg),
        ('Figure 4: Re vs RMSD', generate_figure4_re),
        ('Figure 4: Diversity Heatmap', generate_figure4_diversity),
    ],
}


def generate_figures(figure_names=None, output_dir=None):
    """
    Generate specified figures or all figures.
    
    Parameters
    ----------
    figure_names : list of str, optional
        List of figure group names to generate. If None, generates all.
    output_dir : Path, optional
        Custom output directory. If None, uses default.
    """
    # Apply standard plot style
    apply_plot_style()
    
    # Determine which figures to generate
    if figure_names is None:
        groups_to_run = FIGURE_GROUPS.keys()
    else:
        groups_to_run = figure_names
    
    # Set output directory
    if output_dir is None:
        output_dir = SCRIPT_OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Count total figures to generate
    total_count = sum(len(FIGURE_GROUPS[g]) for g in groups_to_run if g in FIGURE_GROUPS)
    
    # Generate figures with progress bar
    with tqdm(total=total_count, desc="Generating figures", unit="fig") as pbar:
        for group_name in groups_to_run:
            if group_name not in FIGURE_GROUPS:
                print(f"Warning: Unknown figure group '{group_name}', skipping...")
                continue
            
            for description, func in FIGURE_GROUPS[group_name]:
                pbar.set_description(f"{description:<40}")
                try:
                    func(output_dir=output_dir)
                    pbar.update(1)
                except Exception as e:
                    pbar.write(f"  ERROR in {description}: {e}")
                    pbar.update(1)
                    continue
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully generated figures in {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate all publication figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available figure groups:
  figure2       - Model comparison plots (RMSD, Rg, SAXS, metrics)
  figure3       - Apo-holo analysis plots (P(r), RMSD, recovery, correlation)
  notebook_pr   - Notebook P(r) comparison plots
  figure4       - Ensemble analysis plots (barplot, Rg, Re, diversity)

Examples:
  # Generate all figures
  python generate_all_figures.py
  
  # Generate only Figure 2 and 3
  python generate_all_figures.py --figures figure2 figure3
  
  # Generate to custom output directory
  python generate_all_figures.py --output-dir results/
        """
    )
    
    parser.add_argument(
        '--figures',
        nargs='+',
        choices=list(FIGURE_GROUPS.keys()),
        help='Specific figure groups to generate'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory'
    )
    
    args = parser.parse_args()
    
    generate_figures(
        figure_names=args.figures,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
