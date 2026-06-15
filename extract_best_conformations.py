#!/usr/bin/env python3
"""
Extract best conformation metrics from ensemble analysis and save to CSV.

The 'best' conformation per protein is the one with the lowest RMSD to the
ground truth structure (from among the top-50 SAXS-loss-ranked ensemble members).
Metrics were computed in ensemble_average.csv: rg_best, re_best, rmsd_best, loss_best.

This script:
  1. Reads ensemble_average.csv
  2. Saves a focused best_conformations.csv with best metrics + ground truth
  3. Regenerates Rg/Re plots for both average and best conformations
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from metfish.analysis.analysis_draft import FigureVisualization
from metfish.analysis.plot_config import apply_plot_style

apply_plot_style()

# --- Load data ---
src = Path('publication_figures/ensemble_average.csv')
df = pd.read_csv(src)

# --- Build best_conformations.csv ---
best_cols = [
    'protein_id',
    'rg_ref', 're_ref',
    'rg_best', 're_best', 'rmsd_best', 'loss_best',
    'seq_length_best',
]
df_best = df[best_cols].copy()
df_best = df_best.rename(columns={
    'rg_ref':        'rg_ground_truth_nm',
    're_ref':        're_ground_truth_nm',
    'rg_best':       'rg_best_nm',
    're_best':       're_best_nm',
    'rmsd_best':     'rmsd_best_angstrom',
    'loss_best':     'saxs_loss_best',
    'seq_length_best': 'seq_length',
})

out_csv = Path('best_conformations.csv')
df_best.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}  ({len(df_best)} proteins)')
print(df_best.describe().to_string())

# --- Regenerate plots ---
out_dirs = ['analysis_output_test', 'publication_figures', 'analysis_output']

for out_dir in out_dirs:
    viz = FigureVisualization(output_dir=out_dir)

    # Average plots (updated axis labels)
    fig = viz.plot_rg_ensemble(df, save_path='Rg_ensemble.png')
    plt.close(fig)

    fig = viz.plot_re_ensemble(df, save_path='Re_ensemble.png')
    plt.close(fig)

    # Best plots
    fig = viz.plot_rg_best(df, save_path='Rg_best.png')
    plt.close(fig)

    fig = viz.plot_re_best(df, save_path='Re_best.png')
    plt.close(fig)

    print(f'Saved Rg/Re avg+best plots → {out_dir}/')

print('Done.')
