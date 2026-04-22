#!/usr/bin/env python3
"""
Secondary structure (alpha-helix / beta-sheet) propensity analysis.

For each protein in ensemble_average.csv, computes DSSP-based fraction of
residues in helix (H) and beta-sheet (E) states for:
  - Ground truth reference structure
  - Average across top-50 SAXS-loss-ranked ensemble conformations
  - Best conformation (lowest RMSD to ground truth within the ensemble)

Outputs:
  secondary_structure_analysis.csv  -- per-protein results
  Helix_propensity.png / Beta_propensity.png  -- scatter plots (avg + best)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mdtraj as md
from pathlib import Path
from tqdm import tqdm

from metfish.analysis.analysis_draft import (
    DataPaths, NM_TO_ANGSTROM, FigureVisualization
)
from metfish.analysis.plot_config import apply_plot_style

apply_plot_style()

ENSEMBLE_PATH = DataPaths.ENSEMBLE
REF_PATH      = DataPaths.REF
OUT_DIRS      = ['analysis_output_test', 'publication_figures', 'analysis_output']


def dssp_fractions(traj: md.Trajectory) -> tuple:
    """
    Compute per-frame DSSP and return (helix_fraction, beta_fraction) averaged
    across all frames in the trajectory.
    """
    dssp = md.compute_dssp(traj, simplified=False)   # shape (n_frames, n_residues)
    helix = np.mean(np.sum(dssp == 'H', axis=1) / traj.n_residues)
    beta  = np.mean(np.sum(dssp == 'E', axis=1) / traj.n_residues)
    return float(helix), float(beta)


def analyze_protein(protein_id: str):
    """
    For one protein:
      - load ref PDB  → ground-truth helix/beta
      - load top-50 ensemble  → ensemble-avg helix/beta
      - find best (lowest RMSD-to-ref) in ensemble → best helix/beta
    """
    ref_pdb = REF_PATH / f'{protein_id}.pdb'
    loss_csv = ENSEMBLE_PATH / protein_id / 'saved_structures.csv'

    if not ref_pdb.exists() or not loss_csv.exists():
        return None

    # --- reference ---
    ref_traj = md.load(str(ref_pdb))
    ref_ca   = ref_traj.topology.select('name CA')
    helix_ref, beta_ref = dssp_fractions(ref_traj)

    # --- ensemble top-50 by SAXS loss ---
    loss_pd = pd.read_csv(loss_csv)
    loss_pd = loss_pd[~loss_pd['full_path'].str.contains('/best/')]
    loss_pd = loss_pd.sort_values('loss').head(50).reset_index(drop=True)
    pdb_paths = loss_pd['full_path'].tolist()

    if not pdb_paths:
        return None

    try:
        ens_traj = md.load(pdb_paths)
    except Exception as e:
        print(f'  Warning: could not load ensemble for {protein_id}: {e}')
        return None

    ens_ca = ens_traj.topology.select('name CA')
    if len(ens_ca) != len(ref_ca):
        return None

    # DSSP for whole ensemble at once
    helix_avg, beta_avg = dssp_fractions(ens_traj)

    # Best conformation = lowest RMSD to reference
    rmsds = md.rmsd(ens_traj, ref_traj, 0, ens_ca, ref_ca) * NM_TO_ANGSTROM
    best_idx = int(np.argmin(rmsds))
    best_traj = ens_traj[best_idx]
    helix_best, beta_best = dssp_fractions(best_traj)

    return {
        'protein_id':  protein_id,
        'helix_ref':   helix_ref,
        'beta_ref':    beta_ref,
        'helix_avg':   helix_avg,
        'beta_avg':    beta_avg,
        'helix_best':  helix_best,
        'beta_best':   beta_best,
        'rmsd_best':   float(rmsds[best_idx]),
        'n_ensemble':  len(pdb_paths),
    }


def main():
    # --- Load protein list from ensemble_average.csv ---
    src_csv = Path('publication_figures/ensemble_average.csv')
    proteins = pd.read_csv(src_csv)['protein_id'].tolist()
    print(f'Analyzing {len(proteins)} proteins...')

    rows = []
    for pid in tqdm(proteins, unit='protein'):
        result = analyze_protein(pid)
        if result:
            rows.append(result)
        else:
            print(f'  Skipped: {pid}')

    df = pd.DataFrame(rows)
    out_csv = Path('secondary_structure_analysis.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}  ({len(df)} proteins)')

    # --- Summary stats ---
    print('\n--- Helix propensity ---')
    print(df[['helix_ref', 'helix_avg', 'helix_best']].describe().round(3).to_string())
    print('\n--- Beta-sheet propensity ---')
    print(df[['beta_ref', 'beta_avg', 'beta_best']].describe().round(3).to_string())

    # --- Plots ---
    for out_dir in OUT_DIRS:
        viz = FigureVisualization(output_dir=out_dir)

        fig = viz.plot_ss_scatter(df, ss_type='helix', mode='avg',  save_path='Helix_propensity_avg.png')
        plt.close(fig)
        fig = viz.plot_ss_scatter(df, ss_type='helix', mode='best', save_path='Helix_propensity_best.png')
        plt.close(fig)
        fig = viz.plot_ss_scatter(df, ss_type='beta',  mode='avg',  save_path='Beta_propensity_avg.png')
        plt.close(fig)
        fig = viz.plot_ss_scatter(df, ss_type='beta',  mode='best', save_path='Beta_propensity_best.png')
        plt.close(fig)

        print('Saved Helix/Beta plots → ' + out_dir + '/')

    print('\nDone.')


if __name__ == '__main__':
    main()
