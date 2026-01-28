#!/usr/bin/env python3
"""
Analysis Runner - Individual Figure Generation
===============================================

Each figure has its own independent function that can be tested separately.

PERFORMANCE NOTES:
- Figure 2 & 3: Fast (single structures only)
- Figure 4: Slower (loads ensembles with optimized vectorized RMSD)

Usage:
    # Figure 2: Model comparison (RMSD, Rg by OpenFold accuracy)
    python run_analysis.py figure2_rmsd       # Fast - single structures
    python run_analysis.py figure2_rg         # Fast - single structures
    
    # Figure 3: P(r) divergence analysis
    python run_analysis.py figure3_pr         # Fast - single structures
    
    # Figure 4: Ensemble analysis
    python run_analysis.py figure4_barplot    # Slow - loads all ensembles
    python run_analysis.py figure4_rg         # Fast - uses cached data
    python run_analysis.py figure4_re         # Fast - uses cached data
    python run_analysis.py figure4_diversity  # Slow - pairwise RMSD
    
    # Quick replot (instant - from existing CSV files)
    python run_analysis.py replot_figure2_rmsd
    python run_analysis.py replot_figure2_rg
    python run_analysis.py replot_figure3_pr
    python run_analysis.py replot_figure4_barplot
    python run_analysis.py replot_figure4_rg
    python run_analysis.py replot_figure4_re
    python run_analysis.py replot_figure4_diversity
    
    # Apo-holo pair analysis
    python run_analysis.py apo_holo_rmsd <apo_id> <holo_id>
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metfish.analysis.analysis_draft import (
    DataPaths,
    EnsembleAnalyzer,
    ApoHoloPairAnalyzer,
    FigureVisualization,
    DataManager,
    analyze_structure,
    calc_ensemble_diversity,
    calc_pr_l1_loss,
)

# Default output directory
OUTPUT_DIR = Path("./analysis_output")


# =============================================================================
# QUICK REPLOT FUNCTIONS (from existing CSV files)
# =============================================================================

def replot_figure2_rmsd(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 2 RMSD from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'figure2_metrics.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure2_rmsd() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_rmsd_comparison(df, save_path='figure2_rmsd.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure2_rmsd.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure2_rg(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 2 Rg from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'figure2_metrics.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure2_rg() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_rg_comparison(df, save_path='figure2_rg.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure2_rg.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure2_metrics_barplot(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 2 metrics bar plot from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'figure2_metrics.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure2_metrics_barplot() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    
    # Rename AlphaSAXS to NMR for display
    df['Type'] = df['Type'].replace('AlphaSAXS', 'NMR')
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_metrics_comparison_barplot(df, save_path='figure2_metrics_barplot.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure2_metrics_barplot.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure3_pr(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 3 P(r) divergence from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'figure3_pr_divergence.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure3_pr() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_pr_divergence(df, save_path='figure3_pr.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure3_pr.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure3_rmsd(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 3 RMSD divergence from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'figure3_pr_divergence.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure3_pr() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_rmsd_divergence(df, save_path='figure3_rmsd.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure3_rmsd.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure3_recovery_barplot(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 3 AlphaSAXS recovery bar plot from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'figure3_pr_divergence.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure3_pr() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_alphasaxs_recovery_barplot(df, save_path='figure3_recovery_barplot.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure3_recovery_barplot.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure3_correlation(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 3 correlation from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'figure3_pr_divergence.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure3_pr() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_correlation_scatter(df, save_path='figure3_correlation.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure3_correlation.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure4_barplot(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 4 bar plot from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'ensemble_average.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure4_barplot() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} proteins from {csv_path}")
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_ensemble_barplot(df, baseline=4.55, save_path='Ensemble_average.png')
    plt.close(fig)
    
    output_path = output_dir / 'Ensemble_average.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure4_rg(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 4 Rg scatter from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'ensemble_average.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure4_barplot() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} proteins from {csv_path}")
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_rg_ensemble(df, save_path='Rg_ensemble.png')
    plt.close(fig)
    
    output_path = output_dir / 'Rg_ensemble.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure4_re(csv_path: Path = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 4 Re scatter from existing CSV without re-running analysis."""
    output_dir = Path(output_dir)
    csv_path = csv_path or output_dir / 'ensemble_average.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run generate_figure4_barplot() first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} proteins from {csv_path}")
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_re_ensemble(df, save_path='Re_ensemble.png')
    plt.close(fig)
    
    output_path = output_dir / 'Re_ensemble.png'
    print(f"Replotted: {output_path}")
    return output_path


def replot_figure4_diversity(output_dir: Path = OUTPUT_DIR) -> Path:
    """Replot Figure 4 diversity scatter from existing CSVs without re-running analysis."""
    output_dir = Path(output_dir)
    
    csv_accuracy = output_dir / 'ensemble_average.csv'
    csv_diversity = output_dir / 'protein_rmsd_summary.csv'
    
    if not csv_accuracy.exists():
        raise FileNotFoundError(f"CSV not found: {csv_accuracy}. Run generate_figure4_barplot() first.")
    if not csv_diversity.exists():
        raise FileNotFoundError(f"CSV not found: {csv_diversity}. Run generate_figure4_diversity() first.")
    
    df_accuracy = pd.read_csv(csv_accuracy)
    df_diversity = pd.read_csv(csv_diversity)
    print(f"Loaded {len(df_accuracy)} proteins (accuracy) + {len(df_diversity)} proteins (diversity)")
    
    df_merged = df_accuracy.merge(df_diversity, on='protein_id', how='inner')
    
    viz = FigureVisualization(output_dir)
    fig = viz.plot_diversity_scatter(df_merged, save_path='Ensemble_diversity.png')
    plt.close(fig)
    
    output_path = output_dir / 'Ensemble_diversity.png'
    print(f"Replotted: {output_path}")
    return output_path


# =============================================================================
# FIGURE 2a: RMSD Comparison by OpenFold Accuracy
# =============================================================================

def generate_figure2_rmsd(output_dir: Path = OUTPUT_DIR,
                           pdb_ids: list = None,
                           use_cached: bool = True) -> Path:
    """
    Generate Figure 2 RMSD violin plot: RMSD by OpenFold accuracy category.
    
    Compares AlphaSAXS vs OpenFold performance grouped by OpenFold accuracy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    # Try cached
    df = None
    if use_cached:
        df = dm.load_csv('figure2_metrics.csv')
        if df is not None:
            print(f"Loaded cached data: {len(df)} entries")
    
    # Generate if no cache
    if df is None or df.empty:
        analyzer = ApoHoloPairAnalyzer()
        
        if pdb_ids is None:
            pdb_ids = analyzer.get_pdb_list()
        
        print(f"Calculating metrics for {len(pdb_ids)} proteins...")
        df = analyzer.calculate_metrics(pdb_ids)
        df = analyzer.add_openfold_accuracy(df)
        dm.save_csv(df, 'figure2_metrics.csv')
    
    # Generate figure
    fig = viz.plot_rmsd_comparison(df, save_path='figure2_rmsd.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure2_rmsd.png'
    print(f"Generated: {output_path}")
    return output_path


# =============================================================================
# FIGURE 2b: Rg Comparison by OpenFold Accuracy
# =============================================================================

def generate_figure2_rg(output_dir: Path = OUTPUT_DIR,
                         use_cached: bool = True) -> Path:
    """
    Generate Figure 2 Rg violin plot: Rg accuracy by OpenFold accuracy category.
    
    Requires figure2_metrics.csv (run generate_figure2_rmsd first).
    """
    output_dir = Path(output_dir)
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    df = dm.load_csv('figure2_metrics.csv')
    if df is None:
        raise FileNotFoundError("Run generate_figure2_rmsd first to create figure2_metrics.csv")
    
    fig = viz.plot_rg_comparison(df, save_path='figure2_rg.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure2_rg.png'
    print(f"Generated: {output_path}")
    return output_path


def generate_figure2_metrics_barplot(output_dir: Path = OUTPUT_DIR,
                                      use_cached: bool = True) -> Path:
    """
    Generate Figure 2 metrics comparison bar plot: Comparison of all metrics as percentage.
    
    Shows RMSD, SAXS L1 Loss, and Rg Diff comparing AlphaSAXS (NMR) vs OpenFold.
    Loads data from pre-computed model_comparisons.csv.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    # Load from pre-computed CSV
    csv_path = '/global/cfs/cdirs/m4704/100125_Nature_Com_data/results/model_comparisons.csv'
    print(f"Loading pre-computed metrics from {csv_path}...")
    result = pd.read_csv(csv_path)
    
    # Filter for AF and NMR comparisons with target
    AF_result = result[(result['type_a']=='out_AF')&(result['type_b']=='target')][['name','rmsd','saxs_l1','rg_diff']]
    AF_result['rg_diff_A'] = AF_result['rg_diff'] * 10
    AF_result['type'] = 'OpenFold'
    
    NMR_result = result[(result['type_a']=='out_NMR')&(result['type_b']=='target')][['name','rmsd','saxs_l1','rg_diff']]
    NMR_result['rg_diff_A'] = NMR_result['rg_diff'] * 10
    NMR_result['type'] = 'NMR'
    
    # Combine
    df_plot = pd.concat([AF_result, NMR_result])
    print(f"Loaded {len(df_plot)} entries ({len(AF_result)} AF, {len(NMR_result)} NMR)")
    
    # Generate figure
    fig = viz.plot_metrics_comparison_barplot(df_plot, save_path='figure2_metrics_barplot.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure2_metrics_barplot.png'
    print(f"Generated: {output_path}")
    return output_path


# =============================================================================
# FIGURE 3: P(r) Divergence Analysis
# =============================================================================

def generate_figure3_pr(output_dir: Path = OUTPUT_DIR,
                         use_cached: bool = True) -> Path:
    """
    Generate Figure 3: P(r) divergence between apo-holo pairs.
    
    Adds apo-holo similarity categories based on P(r) divergence.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    # Load or create base metrics
    df = dm.load_csv('figure2_metrics.csv')
    if df is None:
        print("Creating figure2 metrics first...")
        generate_figure2_rmsd(output_dir, use_cached=False)
        df = dm.load_csv('figure2_metrics.csv')
    
    # Check if PR divergence already added
    if 'ref_pr_div' not in df.columns:
        print("Adding P(r) divergence...")
        analyzer = ApoHoloPairAnalyzer()
        df = analyzer.add_pr_divergence(df)
        dm.save_csv(df, 'figure3_pr_divergence.csv')
    else:
        dm.save_csv(df, 'figure3_pr_divergence.csv')
    
    # Create violin plot: P(r) Div by apo_holo_similarity
    fig = viz.plot_pr_divergence(df, save_path='figure3_pr.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure3_pr.png'
    print(f"Generated: {output_path}")
    return output_path


def generate_figure3_rmsd(output_dir: Path = OUTPUT_DIR,
                          use_cached: bool = True) -> Path:
    """
    Generate Figure 3 RMSD: RMSD divergence between apo-holo pairs.
    
    Uses the same data as figure3_pr but plots RMSD divergence instead.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    # Load cached data (must have been generated by figure3_pr)
    df = dm.load_csv('figure3_pr_divergence.csv')
    if df is None:
        print("Creating figure3 data first...")
        generate_figure3_pr(output_dir, use_cached=False)
        df = dm.load_csv('figure3_pr_divergence.csv')
    
    # Generate RMSD divergence figure
    fig = viz.plot_rmsd_divergence(df, save_path='figure3_rmsd.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure3_rmsd.png'
    print(f"Generated: {output_path}")
    return output_path


def generate_figure3_recovery_barplot(output_dir: Path = OUTPUT_DIR,
                                       use_cached: bool = True) -> Path:
    """
    Generate Figure 3 AlphaSAXS recovery bar plot.
    
    Shows how well AlphaSAXS recovers apo-holo differences compared to ground truth.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    # Load cached data
    df = dm.load_csv('figure3_pr_divergence.csv')
    if df is None:
        print("Creating figure3 data first...")
        generate_figure3_pr(output_dir, use_cached=False)
        df = dm.load_csv('figure3_pr_divergence.csv')
    
    # Generate recovery bar plot
    fig = viz.plot_alphasaxs_recovery_barplot(df, save_path='figure3_recovery_barplot.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure3_recovery_barplot.png'
    print(f"Generated: {output_path}")
    return output_path


def generate_figure3_correlation(output_dir: Path = OUTPUT_DIR,
                                  use_cached: bool = True) -> Path:
    """
    Generate Figure 3 correlation scatter plot.
    
    Shows correlation between input and generated apo-holo SAXS differences.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    # Load cached data
    df = dm.load_csv('figure3_pr_divergence.csv')
    if df is None:
        print("Creating figure3 data first...")
        generate_figure3_pr(output_dir, use_cached=False)
        df = dm.load_csv('figure3_pr_divergence.csv')
    
    # Generate correlation plot
    fig = viz.plot_correlation_scatter(df, save_path='figure3_correlation.png')
    plt.close(fig)
    
    output_path = output_dir / 'figure3_correlation.png'
    print(f"Generated: {output_path}")
    return output_path


# =============================================================================
# FIGURE 4a: Ensemble Bar Plot (Best vs Average RMSD)
# =============================================================================

def generate_figure4_barplot(output_dir: Path = OUTPUT_DIR, 
                              pdb_ids: list = None,
                              use_cached: bool = True) -> Path:
    """
    Generate Figure 4 bar plot: Best RMSD vs Average RMSD.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for figures
    pdb_ids : list, optional
        List of protein IDs. If None, uses ALL proteins.
    use_cached : bool
        If True, load from ensemble_average.csv if exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    # Try to load cached data
    df = None
    if use_cached:
        df = dm.load_csv('ensemble_average.csv')
        if df is not None:
            print(f"Loaded cached data: {len(df)} proteins")
    
    # Generate if no cache
    if df is None or df.empty:
        if pdb_ids is None:
            pdb_ids = [f[:-4] for f in os.listdir(DataPaths.REF) if f.endswith('.pdb')]
        
        print(f"Analyzing {len(pdb_ids)} proteins...")
        analyzer = EnsembleAnalyzer()
        df, _ = analyzer.analyze_all(pdb_ids)
        dm.save_csv(df, 'ensemble_average.csv')
    
    # Generate figure
    fig = viz.plot_ensemble_barplot(df, baseline=4.55, save_path='Ensemble_average.png')
    plt.close(fig)
    
    output_path = output_dir / 'Ensemble_average.png'
    print(f"Generated: {output_path}")
    return output_path


# =============================================================================
# FIGURE 4b: Rg Scatter Plot
# =============================================================================

def generate_figure4_rg(output_dir: Path = OUTPUT_DIR,
                        use_cached: bool = True) -> Path:
    """
    Generate Figure 4 Rg scatter: Ensemble Rg vs Reference Rg.
    
    Requires ensemble_average.csv (run generate_figure4_barplot first).
    """
    output_dir = Path(output_dir)
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    df = dm.load_csv('ensemble_average.csv')
    if df is None:
        raise FileNotFoundError("Run generate_figure4_barplot first to create ensemble_average.csv")
    
    fig = viz.plot_rg_ensemble(df, save_path='Rg_ensemble.png')
    plt.close(fig)
    
    output_path = output_dir / 'Rg_ensemble.png'
    print(f"Generated: {output_path}")
    return output_path


# =============================================================================
# FIGURE 4c: Re (End-to-End) Scatter Plot
# =============================================================================

def generate_figure4_re(output_dir: Path = OUTPUT_DIR,
                        use_cached: bool = True) -> Path:
    """
    Generate Figure 4 Re scatter: Ensemble Re vs Reference Re.
    
    Requires ensemble_average.csv (run generate_figure4_barplot first).
    """
    output_dir = Path(output_dir)
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    df = dm.load_csv('ensemble_average.csv')
    if df is None:
        raise FileNotFoundError("Run generate_figure4_barplot first to create ensemble_average.csv")
    
    fig = viz.plot_re_ensemble(df, save_path='Re_ensemble.png')
    plt.close(fig)
    
    output_path = output_dir / 'Re_ensemble.png'
    print(f"Generated: {output_path}")
    return output_path


# =============================================================================
# FIGURE 4d: Diversity Scatter Plot
# =============================================================================

def generate_figure4_diversity(output_dir: Path = OUTPUT_DIR,
                                pdb_ids: list = None,
                                use_cached: bool = True) -> Path:
    """
    Generate Figure 4 diversity scatter: Diversity vs Accuracy.
    
    Requires ensemble_average.csv for accuracy data.
    """
    output_dir = Path(output_dir)
    dm = DataManager(output_dir)
    viz = FigureVisualization(output_dir)
    
    # Load accuracy data
    df_accuracy = dm.load_csv('ensemble_average.csv')
    if df_accuracy is None:
        raise FileNotFoundError("Run generate_figure4_barplot first")
    
    # Load or compute diversity
    df_diversity = dm.load_csv('protein_rmsd_summary.csv')
    
    if df_diversity is None:
        if pdb_ids is None:
            pdb_ids = df_accuracy['protein_id'].tolist()
        
        print(f"Computing diversity for {len(pdb_ids)} proteins...")
        analyzer = EnsembleAnalyzer()
        df_diversity = analyzer.analyze_diversity(pdb_ids)
        dm.save_csv(df_diversity, 'protein_rmsd_summary.csv')
    
    # Merge and plot
    df_merged = df_accuracy.merge(df_diversity, on='protein_id', how='inner')
    
    fig = viz.plot_diversity_scatter(df_merged, save_path='Ensemble_diversity.png')
    plt.close(fig)
    
    output_path = output_dir / 'Ensemble_diversity.png'
    print(f"Generated: {output_path}")
    return output_path


# =============================================================================
# FIGURE 4e: Apo-Holo RMSD Scatter
# =============================================================================

def generate_apo_holo_rmsd(apo_id: str, holo_id: str,
                           output_dir: Path = OUTPUT_DIR) -> Path:
    """
    Generate apo-holo RMSD scatter for a specific pair.
    
    Parameters
    ----------
    apo_id : str
        Apo protein ID (e.g., '1JFJ-3_A')
    holo_id : str
        Holo protein ID (e.g., '1JFK_A')
    """
    import mdtraj as md
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz = FigureVisualization(output_dir)
    analyzer = EnsembleAnalyzer()
    
    def get_ensemble_rmsd(pdb_id: str, pair_id: str):
        """Get RMSD to both self and pair reference."""
        ref_path = DataPaths.REF / f'{pdb_id}.pdb'
        pair_path = DataPaths.REF / f'{pair_id}.pdb'
        
        ref_metrics, _, ref_traj = analyze_structure(str(ref_path))
        _, _, pair_traj = analyze_structure(str(pair_path))
        
        ref_ca = ref_traj.topology.select('name CA')
        pair_ca = pair_traj.topology.select('name CA')
        
        ensemble_df = analyzer.extract_ensemble(pdb_id)
        
        results = []
        for _, row in ensemble_df.iterrows():
            try:
                _, _, traj = analyze_structure(row['full_path'])
                traj_ca = traj.topology.select('name CA')
                
                if len(traj_ca) != len(ref_ca):
                    continue
                
                rmsd = md.rmsd(traj, ref_traj, 0, traj_ca, ref_ca)[0] * 10
                rmsd_pair = md.rmsd(traj, pair_traj, 0, traj_ca, pair_ca)[0] * 10
                
                results.append({'rmsd': rmsd, 'rmsd_pair': rmsd_pair})
            except:
                continue
        
        return results
    
    print(f"Analyzing {apo_id}...")
    apo_results = get_ensemble_rmsd(apo_id, holo_id)
    
    print(f"Analyzing {holo_id}...")
    holo_results = get_ensemble_rmsd(holo_id, apo_id)
    
    fig = viz.plot_ensemble_rmsd_scatter(apo_results, holo_results,
                                          save_path=f'RMSD_{apo_id}_vs_{holo_id}.png')
    plt.close(fig)
    
    output_path = output_dir / f'RMSD_{apo_id}_vs_{holo_id}.png'
    print(f"Generated: {output_path}")
    return output_path


# =============================================================================
# CLI Interface
# =============================================================================

COMMANDS = {
    # Figure 2
    'figure2_rmsd': generate_figure2_rmsd,
    'figure2_rg': generate_figure2_rg,
    'figure2_metrics_barplot': generate_figure2_metrics_barplot,
    # Figure 3
    'figure3_pr': generate_figure3_pr,
    'figure3_rmsd': generate_figure3_rmsd,
    'figure3_recovery_barplot': generate_figure3_recovery_barplot,
    'figure3_correlation': generate_figure3_correlation,
    # Figure 4
    'figure4_barplot': generate_figure4_barplot,
    'figure4_rg': generate_figure4_rg,
    'figure4_re': generate_figure4_re,
    'figure4_diversity': generate_figure4_diversity,
    # Replot functions (quick, no analysis)
    'replot_figure2_rmsd': replot_figure2_rmsd,
    'replot_figure2_rg': replot_figure2_rg,
    'replot_figure2_metrics_barplot': replot_figure2_metrics_barplot,
    'replot_figure3_pr': replot_figure3_pr,
    'replot_figure3_rmsd': replot_figure3_rmsd,
    'replot_figure3_recovery_barplot': replot_figure3_recovery_barplot,
    'replot_figure3_correlation': replot_figure3_correlation,
    'replot_figure4_barplot': replot_figure4_barplot,
    'replot_figure4_rg': replot_figure4_rg,
    'replot_figure4_re': replot_figure4_re,
    'replot_figure4_diversity': replot_figure4_diversity,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <command>")
        print("\nFigure 2 (Model Comparison):")
        print("  figure2_rmsd              - RMSD violin by OpenFold accuracy")
        print("  figure2_rg                - Rg violin by OpenFold accuracy")
        print("  figure2_metrics_barplot   - Metrics comparison bar plot")
        print("\nFigure 3 (Apo-Holo Analysis):")
        print("  figure3_pr                - P(r) divergence by apo-holo similarity")
        print("  figure3_rmsd              - RMSD divergence by apo-holo similarity")
        print("  figure3_recovery_barplot  - AlphaSAXS recovery of apo-holo differences")
        print("  figure3_correlation       - Correlation between input and generated SAXS")
        print("\nFigure 4 (Ensemble Analysis):")
        print("  figure4_barplot           - Best vs Average RMSD bar plot")
        print("  figure4_rg                - Rg scatter (ensemble vs reference)")
        print("  figure4_re                - Re scatter (ensemble vs reference)")
        print("  figure4_diversity         - Diversity vs accuracy scatter")
        print("\nQuick Replot (from existing CSV):")
        print("  replot_figure2_rmsd              - Regenerate Figure 2 RMSD plot")
        print("  replot_figure2_rg                - Regenerate Figure 2 Rg plot")
        print("  replot_figure2_metrics_barplot   - Regenerate Figure 2 metrics bar plot")
        print("  replot_figure3_pr                - Regenerate Figure 3 P(r) plot")
        print("  replot_figure3_rmsd              - Regenerate Figure 3 RMSD plot")
        print("  replot_figure3_recovery_barplot  - Regenerate Figure 3 recovery bar plot")
        print("  replot_figure3_correlation       - Regenerate Figure 3 correlation plot")
        print("  replot_figure4_barplot           - Regenerate Figure 4 bar plot")
        print("  replot_figure4_rg                - Regenerate Figure 4 Rg scatter")
        print("  replot_figure4_re                - Regenerate Figure 4 Re scatter")
        print("  replot_figure4_diversity         - Regenerate Figure 4 diversity scatter")
        print("\nApo-Holo Pair:")
        print("  apo_holo_rmsd <apo_id> <holo_id>")
        print("\nRun All:")
        print("  all")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'all':
        print("Running all figures...")
        for name, func in COMMANDS.items():
            print(f"\n--- {name} ---")
            try:
                func()
            except Exception as e:
                print(f"ERROR: {e}")
        print("\nAll figures generated!")
    
    elif command == 'apo_holo_rmsd':
        if len(sys.argv) < 4:
            print("Usage: python run_analysis.py apo_holo_rmsd <apo_id> <holo_id>")
            print("Example: python run_analysis.py apo_holo_rmsd 1JFJ-3_A 1JFK_A")
            sys.exit(1)
        generate_apo_holo_rmsd(sys.argv[2], sys.argv[3])
    
    elif command in COMMANDS:
        COMMANDS[command]()
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
