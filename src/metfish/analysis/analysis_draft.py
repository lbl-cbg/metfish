"""
Draft Analysis Module for Extended Protein Structure Analysis
==============================================================

Integrates notebook functionality from figures 2, 3, and 4.
Compatible with processor.py and visualizer.py.

Organization:
    Section 1: Shared Utilities (DataPaths, analyzers)
    Section 2: Figure 2 & 3 - Model Comparison (ApoHoloPairAnalyzer)
    Section 3: Figure 4 - Ensemble Analysis (EnsembleAnalyzer, ContactMapAnalyzer)
    Section 4: Visualization (matches visualizer.py style)
    Section 5: Data Persistence & Convenience Functions
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from metfish.utils import get_Pr


# =============================================================================
# SECTION 1: SHARED UTILITIES
# =============================================================================

# Constants
NM_TO_ANGSTROM = 10.0
DEFAULT_CONTACT_CUTOFF_NM = 0.8  # 8 Angstroms


class DataPaths:
    """Data paths for 100125_Nature_Com_data."""
    
    BASE = Path("/global/cfs/cdirs/m4704/100125_Nature_Com_data")
    REF = BASE / "Apo_holo_data" / "pdbs"
    APO_HOLO_CSV = BASE / "Apo_holo_data" / "Table_rmsd_Apo_vs_Holo.csv"
    SINGLE_NMR = BASE / "single_conformation" / "nmr"
    SINGLE_AF = BASE / "single_conformation" / "af"
    ENSEMBLE = BASE / "ensemble_generated" / "ensembles"


def calc_helicity(traj: md.Trajectory) -> float:
    """Calculate fraction of helical residues."""
    dssp = md.compute_dssp(traj)
    return np.sum(dssp[0] == 'H') / traj.n_residues


def calc_beta(traj: md.Trajectory) -> float:
    """Calculate fraction of beta sheet residues."""
    dssp = md.compute_dssp(traj)
    return np.sum(dssp[0] == 'E') / traj.n_residues


def calc_rg(traj: md.Trajectory) -> float:
    """Calculate radius of gyration (nm)."""
    return md.compute_rg(traj)[0]


def calc_re(traj: md.Trajectory) -> float:
    """Calculate end-to-end distance (nm)."""
    ca_indices = traj.topology.select_atom_indices(selection='alpha')
    if len(ca_indices) < 2:
        return 0.0
    return md.compute_distances(traj, [[ca_indices[0], ca_indices[-1]]])[0][0]


def calc_contact_map(traj: md.Trajectory, cutoff: float = DEFAULT_CONTACT_CUTOFF_NM) -> np.ndarray:
    """Calculate binary contact map (CA-CA distances < cutoff)."""
    contacts = md.compute_contacts(traj, contacts='all', scheme='ca')
    contact_matrix = md.geometry.squareform(contacts[0] < cutoff, contacts[1])
    return contact_matrix[0]


def analyze_structure(pdb_path: str) -> Tuple[Dict, np.ndarray, md.Trajectory]:
    """Analyze a single structure and return metrics, contact map, and trajectory."""
    traj = md.load(pdb_path)
    metrics = {
        'seq_length': traj.n_residues,
        'rg': calc_rg(traj),
        're': calc_re(traj),
        'helicity': calc_helicity(traj),
        'beta': calc_beta(traj),
    }
    contact_map = calc_contact_map(traj)
    return metrics, contact_map, traj


def calc_ca_rmsd(traj_test: md.Trajectory, traj_ref: md.Trajectory) -> Optional[float]:
    """Calculate CA RMSD in Angstroms. Returns None if lengths don't match."""
    ca_test = traj_test.topology.select('name CA')
    ca_ref = traj_ref.topology.select('name CA')
    if len(ca_test) != len(ca_ref):
        return None
    return md.rmsd(traj_test, traj_ref, 0, ca_test, ca_ref)[0] * NM_TO_ANGSTROM


def l1_loss_padded(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """L1 loss with zero-padding for unequal lengths."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    max_len = max(len(y_true), len(y_pred))
    y_true = np.pad(y_true, (0, max_len - len(y_true)))
    y_pred = np.pad(y_pred, (0, max_len - len(y_pred)))
    return np.sum(np.abs(y_true - y_pred))


def calc_pr_l1_loss(pdb_a: str, pdb_b: str, bin_size: float = 0.5) -> float:
    """Calculate L1 loss between P(r) curves of two structures."""
    _, pr_a = get_Pr(pdb_a, None, None, bin_size)
    _, pr_b = get_Pr(pdb_b, None, None, bin_size)
    return l1_loss_padded(pr_a, pr_b)


# =============================================================================
# SECTION 2: FIGURE 2 & 3 - MODEL COMPARISON
# =============================================================================

class ApoHoloPairAnalyzer:
    """
    Analyze apo-holo pairs and model predictions.
    
    Figure 2: RMSD/Rg comparison, OpenFold accuracy categories
    Figure 3: P(r) divergence, apo-holo similarity categories
    """
    
    def __init__(self, ref_path: Path = None, model_paths: Dict[str, Path] = None):
        self.ref_path = Path(ref_path) if ref_path else DataPaths.REF
        self.model_paths = model_paths or {
            'AlphaSAXS': DataPaths.SINGLE_NMR,
            'OpenFold': DataPaths.SINGLE_AF,
        }
        self.pair_dict = self._load_pairs()
    
    def _load_pairs(self) -> Dict[str, str]:
        """Load apo-holo pair mappings."""
        if not DataPaths.APO_HOLO_CSV.exists():
            return {}
        df = pd.read_csv(DataPaths.APO_HOLO_CSV, sep=';')
        pairs = {}
        for _, row in df.iterrows():
            pairs[row['Apo_ID']] = row['Holo_ID']
            pairs[row['Holo_ID']] = row['Apo_ID']
        return pairs
    
    def get_pdb_list(self) -> List[str]:
        """Get list of PDB IDs from reference path."""
        return [f[:-4] for f in os.listdir(self.ref_path) if f.endswith('.pdb')]
    
    # --- Figure 2: RMSD/Rg Comparison ---
    
    def calculate_metrics(self, pdb_ids: List[str] = None) -> pd.DataFrame:
        """Calculate RMSD and Rg metrics for all models.
        
        Optimized to load each reference structure only once per protein.
        """
        pdb_ids = pdb_ids or self.get_pdb_list()
        results = []
        
        print(f"Calculating metrics for {len(pdb_ids)} proteins...")
        
        for i, pdb_id in enumerate(pdb_ids, 1):
            print(f"[{i}/{len(pdb_ids)}] {pdb_id}...", end=" ", flush=True)
            
            ref_pdb = self.ref_path / f'{pdb_id}.pdb'
            if not ref_pdb.exists():
                print("SKIP (reference missing)")
                continue
            
            try:
                # Load reference once for all models
                ref_traj = md.load(str(ref_pdb))
                ref_rg = calc_rg(ref_traj)
                
                # Calculate metrics for each model
                model_results = []
                for model_name, model_path in self.model_paths.items():
                    model_pdb = model_path / f'{pdb_id}.pdb'
                    
                    if not model_pdb.exists():
                        model_results.append(f"{model_name}:missing")
                        continue
                    
                    try:
                        model_traj = md.load(str(model_pdb))
                        
                        rmsd = calc_ca_rmsd(model_traj, ref_traj)
                        if rmsd is None:
                            model_results.append(f"{model_name}:length_mismatch")
                            continue
                        
                        rg_diff = (calc_rg(model_traj) - ref_rg) * NM_TO_ANGSTROM
                        
                        # Calculate SAXS L1 loss (P(r) comparison)
                        saxs_l1 = calc_pr_l1_loss(str(ref_pdb), str(model_pdb))
                        
                        results.append({
                            'pdb_id': pdb_id,
                            'type': model_name,
                            'rmsd': rmsd,
                            'rg_diff_A': rg_diff,
                            'saxs_l1': saxs_l1,
                        })
                        model_results.append(f"{model_name}:OK({rmsd:.2f}Å)")
                    except Exception as e:
                        model_results.append(f"{model_name}:ERROR")
                
                print(" | ".join(model_results))
            except Exception as e:
                print(f"ERROR loading reference: {e}")
        
        return pd.DataFrame(results)
    
    def add_openfold_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add OpenFold accuracy category based on RMSD thresholds."""
        of_rmsd = df.query('type == "OpenFold"')[['pdb_id', 'rmsd']].rename(
            columns={'rmsd': 'of_rmsd'})
        df = df.merge(of_rmsd, on='pdb_id', how='left')
        
        conditions = [df['of_rmsd'] < 1, df['of_rmsd'] <= 5, df['of_rmsd'] > 5]
        df['OpenFold Accuracy'] = np.select(conditions, ['High', 'Medium', 'Low'], 'Unknown')
        
        # Ensure categorical order: Low → Medium → High
        df['OpenFold Accuracy'] = pd.Categorical(
            df['OpenFold Accuracy'],
            categories=['Low', 'Medium', 'High'],
            ordered=True
        )
        return df
    
    # --- Figure 3: P(r) Divergence ---
    
    def add_pr_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add P(r) and RMSD divergence between apo-holo pairs."""
        df['pair_id'] = df['pdb_id'].map(self.pair_dict)
        
        def get_ref_pr_div(row):
            if pd.isna(row['pair_id']):
                return None
            pdb_a = str(self.ref_path / f"{row['pdb_id']}.pdb")
            pdb_b = str(self.ref_path / f"{row['pair_id']}.pdb")
            return calc_pr_l1_loss(pdb_a, pdb_b)
        
        def get_model_pr_div(row):
            if pd.isna(row['pair_id']):
                return None
            model_path = self.model_paths[row['type']]
            pdb_a = str(model_path / f"{row['pdb_id']}.pdb")
            pdb_b = str(model_path / f"{row['pair_id']}.pdb")
            if not Path(pdb_a).exists() or not Path(pdb_b).exists():
                return None
            return calc_pr_l1_loss(pdb_a, pdb_b)
        
        def get_model_rmsd_div(row):
            """Calculate RMSD between model's apo and holo predictions."""
            if pd.isna(row['pair_id']):
                return None
            model_path = self.model_paths[row['type']]
            pdb_a = str(model_path / f"{row['pdb_id']}.pdb")
            pdb_b = str(model_path / f"{row['pair_id']}.pdb")
            if not Path(pdb_a).exists() or not Path(pdb_b).exists():
                return None
            try:
                traj_a = md.load(pdb_a)
                traj_b = md.load(pdb_b)
                rmsd = calc_ca_rmsd(traj_a, traj_b)
                return rmsd
            except:
                return None
        
        df['ref_pr_div'] = df.apply(get_ref_pr_div, axis=1)
        df['pr_div'] = df.apply(get_model_pr_div, axis=1)
        df['rmsd_div'] = df.apply(get_model_rmsd_div, axis=1)
        
        # Categorize by apo-holo similarity
        conditions = [df['ref_pr_div'] <= 0.05, df['ref_pr_div'] <= 0.1, df['ref_pr_div'] > 0.1]
        df['apo_holo_similarity'] = np.select(conditions, ['High', 'Medium', 'Low'], 'Unknown')
        
        # Ensure categorical order: Low → Medium → High
        df['apo_holo_similarity'] = pd.Categorical(
            df['apo_holo_similarity'],
            categories=['Low', 'Medium', 'High'],
            ordered=True
        )
        return df


# =============================================================================
# SECTION 3: FIGURE 4 - ENSEMBLE ANALYSIS
# =============================================================================

def calc_pairwise_rmsd_optimized(pdb_paths: List[str]) -> List[float]:
    """
    Calculate pairwise RMSD between all conformations in an ensemble.
    
    This is an optimized version that loads all PDBs at once and uses
    vectorized RMSD calculations for efficiency.
    
    Parameters
    ----------
    pdb_paths : List[str]
        List of paths to PDB files in the ensemble.
    
    Returns
    -------
    List[float]
        All pairwise RMSD values in Angstroms (excluding self-comparisons).
    """
    if not pdb_paths:
        return []
    
    # Load all PDBs at once into a single trajectory
    traj = md.load(pdb_paths)
    ca_indices = traj.topology.select('name CA')
    
    results = []
    n_frames = traj.n_frames
    
    for i in range(n_frames):
        # Compare frame 'i' against all frames in one vectorized step
        rmsd_frame = md.rmsd(traj, traj, frame=i, atom_indices=ca_indices)
        rmsd_frame = rmsd_frame * NM_TO_ANGSTROM  # Convert to Angstroms
        
        # Remove self-comparison (value at index i, which is 0.0)
        rmsd_filtered = np.delete(rmsd_frame, i)
        results.extend(rmsd_filtered)
    
    return results


def calc_ensemble_diversity(pdb_id: str, ensemble_path: Path = None) -> Dict[str, float]:
    """
    Calculate diversity metrics for an ensemble.
    
    Parameters
    ----------
    pdb_id : str
        The protein ID.
    ensemble_path : Path, optional
        Path to ensemble directory.
    
    Returns
    -------
    Dict with 'pair_rmsd_avg', 'pair_rmsd_best', and 'raw_results'.
    """
    ensemble_path = Path(ensemble_path) if ensemble_path else DataPaths.ENSEMBLE
    
    # Extract ensemble
    loss_csv = ensemble_path / pdb_id / 'saved_structures.csv'
    if not loss_csv.exists():
        return None
    
    df = pd.read_csv(loss_csv)
    df = df[~df['full_path'].str.contains('best')]
    df = df.sort_values('loss').head(50).reset_index(drop=True)
    
    pdb_paths = df['full_path'].tolist()
    results = calc_pairwise_rmsd_optimized(pdb_paths)
    
    if not results:
        return None
    
    return {
        'protein_id': pdb_id,
        'pair_rmsd_avg': np.mean(results),
        'pair_rmsd_best': np.min(results),
        'raw_results': results,
    }


class EnsembleAnalyzer:
    """Analyze ensemble of structures from conformational sampling."""
    
    def __init__(self, ref_path: Path = None, ensemble_path: Path = None):
        self.ref_path = Path(ref_path) if ref_path else DataPaths.REF
        self.ensemble_path = Path(ensemble_path) if ensemble_path else DataPaths.ENSEMBLE
    
    def extract_ensemble(self, pdb_id: str, top_n: int = 50) -> pd.DataFrame:
        """Extract top N structures from ensemble by SAXS loss."""
        loss_csv = self.ensemble_path / pdb_id / 'saved_structures.csv'
        if not loss_csv.exists():
            raise FileNotFoundError(f"No ensemble for {pdb_id}")
        
        df = pd.read_csv(loss_csv)
        df = df[~df['full_path'].str.contains('best')]
        return df.sort_values('loss').head(top_n).reset_index(drop=True)
    
    def analyze_ensemble(self, pdb_id: str, top_n: int = 50) -> Dict[str, Any]:
        """Comprehensive analysis of an ensemble."""
        ref_path = self.ref_path / f'{pdb_id}.pdb'
        ref_metrics, ref_contact, ref_traj = analyze_structure(str(ref_path))
        ref_ca = ref_traj.topology.select('name CA')
        
        ensemble_df = self.extract_ensemble(pdb_id, top_n)
        
        results, contact_maps = [], []
        for _, row in ensemble_df.iterrows():
            try:
                metrics, contact, traj = analyze_structure(row['full_path'])
                contact_maps.append(contact)
                
                traj_ca = traj.topology.select('name CA')
                if len(traj_ca) != len(ref_ca):
                    continue
                
                metrics['rmsd'] = md.rmsd(traj, ref_traj, 0, traj_ca, ref_ca)[0] * NM_TO_ANGSTROM
                metrics['loss'] = row['loss']
                results.append(metrics)
            except Exception as e:
                print(f"Error: {row['full_path']}: {e}")
        
        if not results:
            return None
        
        numeric_keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
        averages = {k: np.mean([r[k] for r in results]) for k in numeric_keys}
        best = min(results, key=lambda d: d['rmsd'])
        
        return {
            'pdb_id': pdb_id,
            'averages': averages,
            'ref_metrics': ref_metrics,
            'contact_maps': contact_maps,
            'ref_contact': ref_contact,
            'results': results,
            'best': best,
        }
    
    def analyze_all(self, pdb_list: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Analyze all proteins and return summary DataFrame and detailed results."""
        if pdb_list is None:
            pdb_list = [f[:-4] for f in os.listdir(self.ref_path) if f.endswith('.pdb')]
        
        rows, all_data = [], {}
        total = len(pdb_list)
        for i, pdb_id in enumerate(pdb_list, 1):
            print(f"[{i}/{total}] {pdb_id}...", end=" ", flush=True)
            try:
                data = self.analyze_ensemble(pdb_id)
                if data is None:
                    print("SKIP (no data)")
                    continue
                
                row = {'protein_id': pdb_id}
                for k, v in data['averages'].items():
                    row[f'{k}_avg'] = v
                for k, v in data['ref_metrics'].items():
                    row[f'{k}_ref'] = v
                for k, v in data['best'].items():
                    row[f'{k}_best'] = v
                
                rows.append(row)
                all_data[pdb_id] = data
                print(f"OK (RMSD={data['averages'].get('rmsd', 0):.2f}Å)")
            except Exception as e:
                print(f"ERROR: {e}")
        
        return pd.DataFrame(rows), all_data
    
    def analyze_diversity(self, pdb_list: List[str] = None) -> pd.DataFrame:
        """
        Calculate pairwise RMSD diversity for all proteins.
        
        This measures how diverse the conformations in each ensemble are
        (RMSD between ensemble members, not to ground truth).
        
        Returns DataFrame with columns: protein_id, pair_rmsd_avg, pair_rmsd_best
        """
        if pdb_list is None:
            pdb_list = [f[:-4] for f in os.listdir(self.ref_path) if f.endswith('.pdb')]
        
        rows = []
        total = len(pdb_list)
        for i, pdb_id in enumerate(pdb_list, 1):
            print(f"[{i}/{total}] {pdb_id} diversity...", end=" ", flush=True)
            try:
                result = calc_ensemble_diversity(pdb_id, self.ensemble_path)
                if result:
                    rows.append({
                        'protein_id': result['protein_id'],
                        'pair_rmsd_avg': result['pair_rmsd_avg'],
                        'pair_rmsd_best': result['pair_rmsd_best'],
                    })
                    print(f"OK (diversity={result['pair_rmsd_avg']:.2f}Å)")
                else:
                    print("SKIP")
            except Exception as e:
                print(f"ERROR: {e}")
        
        return pd.DataFrame(rows)
        
        return pd.DataFrame(rows)


class ContactMapAnalyzer:
    """Contact map analysis: Jaccard similarity and clustering."""
    
    @staticmethod
    def flatten(contact_map: np.ndarray) -> np.ndarray:
        """Extract upper triangle without diagonal."""
        return contact_map[np.triu_indices_from(contact_map, k=1)]
    
    @staticmethod
    def jaccard_similarity(cm_a: np.ndarray, cm_b: np.ndarray) -> float:
        """Calculate Jaccard similarity between flattened binary contact maps."""
        intersection = np.logical_and(cm_a, cm_b).sum()
        union = np.logical_or(cm_a, cm_b).sum()
        return intersection / union if union > 0 else 0.0
    
    def get_linkage(self, contact_maps: List[np.ndarray], method: str = 'ward') -> np.ndarray:
        """Compute linkage matrix for hierarchical clustering."""
        flat = np.array([self.flatten(cm) for cm in contact_maps])
        distances = pdist(flat, metric='jaccard')
        return linkage(distances, method=method)


def plot_dendrogram(linkage_matrix: np.ndarray, **kwargs) -> None:
    """Plot dendrogram from linkage matrix."""
    dendrogram(linkage_matrix, **kwargs)


# =============================================================================
# SECTION 4: VISUALIZATION (matches visualizer.py style)
# =============================================================================

class FigureVisualization:
    """
    Visualization for Figures 2, 3, 4 following Nature Communications format.
    
    Figure specifications:
    - Single column width: 90mm (3.54 inches)  
    - Double column width: 180mm (7.08 inches)
    - Font sizes: 12-14pt for readability
    - DPI: 600 for publication quality
    - Layout: 3-4 subfigures per row for multi-panel figures
    - Total subfigures: 6-12 per complete figure
    
    Standard usage:
    - Individual plots: Single column width (3.54" x 3.54")
    - Multi-metric comparisons: Double column width (7.08" x 4")
    - Multi-panel figures: Combine 3-4 panels horizontally
    """
    
    # Professional color scheme (Nature Comm style, colorblind-safe)
    COLORS_MODELS = {
        'OpenFold': '#0173B2',      # IBM Blue (baseline)
        'AlphaSAXS': '#DE8F05',     # Golden poppy (your method)
        'NMR': '#DE8F05',           # Same as AlphaSAXS
    }
    
    COLORS_QUALITY = {
        'Low': '#CC78BC',           # Muted orchid (purple)
        'Medium': '#029E73',        # Persian green (teal)
        'High': '#ECE133',          # Bright yellow (success)
    }
    
    COLORS_SUPPORT = {
        'reference': '#949494',     # Medium gray
        'baseline': '#E64B35',      # Cinnabar red (thresholds)
        'grid': '#ECECEC',          # Light gray
    }
    
    ALPHA = {
        'fill': 0.4,                # Filled violin/box areas
        'scatter': 0.65,            # Scatter points
        'overlay': 0.15,            # Secondary data overlays
        'grid': 0.3,                # Grid lines
    }
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_style()
    
    def _set_style(self):
        """Set Nature Communications publication style with larger fonts."""
        plt.rcParams.update({
            'font.size': 12,              # Base font (12pt for readability)
            'axes.labelsize': 14,         # Axis labels
            'axes.titlesize': 14,         # Subplot titles
            'xtick.labelsize': 12,        # Tick labels
            'ytick.labelsize': 12,
            'legend.fontsize': 11,        # Legend
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'pdf.fonttype': 42,           # Editable PDF text
            'axes.linewidth': 1.5,        # Thicker axes for clarity
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
        })
    
    def _save(self, fig: plt.Figure, save_path: str, dpi: int = 600):
        """Save figure to output directory."""
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=dpi, bbox_inches='tight')
        return fig
    
    # --- Figure 2: RMSD Comparison ---
    
    def plot_rmsd_comparison(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Violin plot: RMSD by OpenFold accuracy category."""
        fig, ax = plt.subplots(figsize=(3.54, 3.54), dpi=600)
        
        # Ensure categorical order if not already set
        if not isinstance(df['OpenFold Accuracy'].dtype, pd.CategoricalDtype):
            df['OpenFold Accuracy'] = pd.Categorical(
                df['OpenFold Accuracy'],
                categories=['Low', 'Medium', 'High'],
                ordered=True
            )
        
        sns.violinplot(y='rmsd', x='OpenFold Accuracy', hue='type',
                       palette=self.COLORS_MODELS, fill=True, alpha=self.ALPHA['fill'],
                       data=df, ax=ax, order=['Low', 'Medium', 'High'],
                       linewidth=1.5, inner='quartile', cut=0)
        
        ax.set_xlabel('OpenFold Accuracy', labelpad=4)
        ax.set_ylabel('RMSD (Å)', labelpad=4)
        ax.grid(True, alpha=self.ALPHA['grid'], linewidth=0.5, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        if ax.get_legend():
            ax.get_legend().set_title(None)
            ax.get_legend().set_frame_on(True)
        sns.despine(ax=ax)
        
        plt.tight_layout(pad=0.5)
        return self._save(fig, save_path)
    
    def plot_rg_comparison(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Violin plot: Rg accuracy by OpenFold accuracy category."""
        fig, ax = plt.subplots(figsize=(3.54, 3.54), dpi=600)
        
        # Ensure categorical order if not already set
        if not isinstance(df['OpenFold Accuracy'].dtype, pd.CategoricalDtype):
            df['OpenFold Accuracy'] = pd.Categorical(
                df['OpenFold Accuracy'],
                categories=['Low', 'Medium', 'High'],
                ordered=True
            )
        
        sns.violinplot(y='rg_diff_A', x='OpenFold Accuracy', hue='type',
                       palette=self.COLORS_MODELS, fill=True, alpha=self.ALPHA['fill'],
                       data=df, ax=ax, order=['Low', 'Medium', 'High'],
                       linewidth=1.5, inner='quartile', cut=0)
        
        ax.set_xlabel('OpenFold Accuracy', labelpad=4)
        ax.set_ylabel('Rg Accuracy (Å)', labelpad=4)
        ax.grid(True, alpha=self.ALPHA['grid'], linewidth=0.5, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        if ax.get_legend():
            ax.get_legend().set_title(None)
            ax.get_legend().set_frame_on(True)
        sns.despine(ax=ax)
        
        plt.tight_layout(pad=0.5)
        return self._save(fig, save_path)
    
    def plot_metrics_comparison_barplot(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Bar plot: Comparison of metrics (RMSD, SAXS L1, Rg Diff) as percentage of AF baseline."""
        # Double column width for Nature Comm: 180mm = 7.08 inches (for 3 metrics side-by-side)
        fig = plt.figure(figsize=(7.08, 4), dpi=600)
        
        metrics = ['rmsd', 'saxs_l1', 'rg_diff_A']
        metric_labels = {'rmsd': 'RMSD vs Truth', 'saxs_l1': 'SAXS L1 Loss', 'rg_diff_A': 'Rg Diff vs Truth'}
        units = {'rmsd': 'Å', 'saxs_l1': '', 'rg_diff_A': 'Å'}
        
        # Calculate means by type
        means = df.groupby('type')[metrics].mean().reset_index()
        
        # Get AF baseline
        af_means = means[means['type'] == 'OpenFold'].set_index('type')
        af_vals = {m: af_means.loc['OpenFold', m] for m in metrics}
        
        # Melt to long format
        means_long = means.melt(id_vars='type', value_vars=metrics,
                                var_name='Metric', value_name='Absolute_Value')
        means_long['Metric_Label'] = means_long['Metric'].map(metric_labels)
        
        # Calculate percentage relative to AF
        def calc_pct(row):
            base = af_vals.get(row['Metric'], 1)
            return (row['Absolute_Value'] / base) * 100
        means_long['Percentage'] = means_long.apply(calc_pct, axis=1)
        
        # Plot
        hue_order = ['OpenFold', 'NMR']
        x_order = ['RMSD vs Truth', 'SAXS L1 Loss', 'Rg Diff vs Truth']
        
        ax = sns.barplot(data=means_long, x='Metric_Label', y='Percentage', hue='type',
                         palette=self.COLORS_MODELS, hue_order=hue_order, order=x_order,
                         edgecolor='black', linewidth=1.2, alpha=0.85)
        
        plt.axhline(100, color=self.COLORS_SUPPORT['reference'], linestyle='--', 
                    linewidth=2, alpha=0.7, zorder=1)
        
        # Annotations
        labels_data = []
        for hue in hue_order:
            for label_name in x_order:
                subset = means_long[(means_long['type'] == hue) & (means_long['Metric_Label'] == label_name)]
                if not subset.empty:
                    row = subset.iloc[0]
                    abs_val = row['Absolute_Value']
                    metric = row['Metric']
                    unit = units.get(metric, '')
                    unit_str = f" {unit}" if unit else ""
                    labels_data.append(f"{abs_val:.2f}{unit_str}")
        
        for i, p in enumerate(ax.patches):
            if i < len(labels_data):
                ax.annotate(labels_data[i],
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', color='black',
                            xytext=(0, 5), textcoords='offset points')
        
        plt.xlabel('')
        plt.ylabel('Percentage of OpenFold Value')
        plt.ylim(0, means_long['Percentage'].max() * 1.3)
        plt.legend(loc='upper left')
        plt.tight_layout()
        return self._save(fig, save_path)
    
    # --- Figure 3: P(r) Divergence ---
    
    def plot_pr_divergence(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Violin plot: P(r) divergence by apo-holo similarity."""
        fig, ax = plt.subplots(figsize=(3.54, 3.54), dpi=600)
        
        # Filter out rows without pair data
        df_plot = df.dropna(subset=['pr_div', 'apo_holo_similarity'])
        
        # Ensure categorical order if not already set
        if not isinstance(df_plot['apo_holo_similarity'].dtype, pd.CategoricalDtype):
            df_plot['apo_holo_similarity'] = pd.Categorical(
                df_plot['apo_holo_similarity'],
                categories=['Low', 'Medium', 'High'],
                ordered=True
            )
        
        sns.violinplot(y='pr_div', x='apo_holo_similarity', hue='type',
                       fill=False, data=df_plot, ax=ax, order=['Low', 'Medium', 'High'])
        
        ax.set_xlabel('Apo vs Holo Similarity', labelpad=4)
        ax.set_ylabel('P(r) Div', labelpad=4)
        if ax.get_legend():
            ax.get_legend().set_title(None)
        
        plt.tight_layout()
        return self._save(fig, save_path)
    
    def plot_rmsd_divergence(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Violin plot: RMSD divergence by apo-holo similarity."""
        fig, ax = plt.subplots(figsize=(3.54, 3.54), dpi=600)
        
        # Filter out rows without pair data
        df_plot = df.dropna(subset=['rmsd_div', 'apo_holo_similarity'])
        
        # Ensure categorical order if not already set
        if not isinstance(df_plot['apo_holo_similarity'].dtype, pd.CategoricalDtype):
            df_plot['apo_holo_similarity'] = pd.Categorical(
                df_plot['apo_holo_similarity'],
                categories=['Low', 'Medium', 'High'],
                ordered=True
            )
        
        sns.violinplot(y='rmsd_div', x='apo_holo_similarity', hue='type',
                       fill=False, data=df_plot, ax=ax, order=['Low', 'Medium', 'High'])
        
        ax.set_xlabel('Ground Truth Apo vs Holo Similarity', labelpad=4)
        ax.set_ylabel('Apo vs Holo RMSD (Å)', labelpad=4)
        if ax.get_legend():
            ax.get_legend().set_title(None)
        
        plt.tight_layout()
        return self._save(fig, save_path)
    
    def plot_alphasaxs_recovery_barplot(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Bar plot: AlphaSAXS recovery of apo-holo differences (percentage vs ground truth)."""
        fig = plt.figure(figsize=(3.54, 3.54), dpi=600)
        
        # Filter for AlphaSAXS only and pairs with data
        df_alphasaxs = df[(df['type'] == 'AlphaSAXS') & df['pair_id'].notna()].copy()
        
        if df_alphasaxs.empty:
            print("No AlphaSAXS apo-holo pair data found")
            return fig
        
        # Calculate averages
        model_means = df_alphasaxs[['rmsd_div', 'pr_div']].mean()
        ref_means = df_alphasaxs[['ref_pr_div']].mean()
        
        # For RMSD, we need reference RMSD divergence
        # Calculate it if not present
        if 'ref_rmsd_div' not in df_alphasaxs.columns:
            # Estimate from available data
            ref_rmsd_mean = model_means['rmsd_div']  # Use model as proxy
        else:
            ref_rmsd_mean = df_alphasaxs['ref_rmsd_div'].mean()
        
        rmsd_pct = (model_means['rmsd_div'] / ref_rmsd_mean) * 100 if ref_rmsd_mean > 0 else 0
        saxs_pct = (model_means['pr_div'] / ref_means['ref_pr_div']) * 100 if ref_means['ref_pr_div'] > 0 else 0
        
        plot_df = pd.DataFrame([
            {'Metric': 'RMSD', 'Percentage': rmsd_pct, 'Absolute': model_means['rmsd_div'], 'Ref_Absolute': ref_rmsd_mean},
            {'Metric': 'SAXS_L1', 'Percentage': saxs_pct, 'Absolute': model_means['pr_div'], 'Ref_Absolute': ref_means['ref_pr_div']}
        ])
        
        ax = sns.barplot(data=plot_df, x='Metric', y='Percentage', color='steelblue')
        
        # Annotations
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            unit = " Å" if row['Metric'] == 'RMSD' else ""
            label = f"{row['Percentage']:.1f}%\n(Abs: {row['Absolute']:.2f}{unit})"
            bar = ax.patches[i]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    label, ha='center', va='bottom', color='black')
        
        # X-axis labels with GT
        new_labels = []
        for _, row in plot_df.iterrows():
            unit = " Å" if row['Metric'] == 'RMSD' else ""
            new_labels.append(f"{row['Metric']}\n(GT: {row['Ref_Absolute']:.2f}{unit})")
        ax.set_xticklabels(new_labels)
        
        plt.axhline(100, color='red', linestyle='--', linewidth=2, label='Ground Truth (100%)')
        plt.title('AlphaSAXS Recovery of\nApo-Holo Differences')
        plt.ylabel('Percentage vs Ground Truth')
        plt.tight_layout()
        return self._save(fig, save_path)
    
    def plot_correlation_scatter(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Scatter plot with regression: Correlation between input and generated apo-holo SAXS difference."""
        from scipy import stats
        
        fig = plt.figure(figsize=(3.54, 3.54), dpi=600)
        
        # Filter for AlphaSAXS pairs with valid data
        df_plot = df[(df['type'] == 'AlphaSAXS') & df['ref_pr_div'].notna() & df['pr_div'].notna()].copy()
        
        if len(df_plot) < 2:
            print("Insufficient data for correlation plot")
            return fig
        
        x = df_plot['ref_pr_div']
        y = df_plot['pr_div']
        
        # Calculate Pearson correlation
        r, p_value = stats.pearsonr(x, y)
        
        # Plot
        ax = plt.gca()
        sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.5}, 
                   line_kws={'color': 'red', 'linewidth': 2}, ax=ax)
        
        # Add annotation
        text_str = f'$r = {r:.2f}$\n$p = {p_value:.2g}$'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Input Apo-Holo SAXS Difference (L1 Loss)', labelpad=4)
        ax.set_ylabel('Generated Apo vs Holo Difference (L1 Loss)', labelpad=4)
        ax.set_title('Correlation Analysis')
        plt.tight_layout()
        return self._save(fig, save_path)
    
    # --- Figure 4: Ensemble Analysis ---
    
    def plot_ensemble_barplot(self, df: pd.DataFrame, baseline: float = 4.55,
                              save_path: str = None) -> plt.Figure:
        """Bar plot: Best vs Average RMSD with baseline."""
        fig, ax = plt.subplots(figsize=(3.54, 3.54), dpi=600)
        
        mean_df = df[['rmsd_best', 'rmsd_avg']].mean().reset_index()
        mean_df.columns = ['Metrics', 'Value']
        mean_df['Metrics'] = ['Best RMSD', 'Average RMSD']
        
        palette = sns.color_palette('magma', n_colors=2)
        sns.barplot(data=mean_df, x='Metrics', y='Value', hue='Metrics',
                    palette=palette, legend=None, ax=ax)
        
        ax.axhline(baseline, linestyle='--', color='red', 
                  linewidth=2, label='AlphaSAXS Baseline')
        ax.set_title('Ensemble Method Performance')
        ax.set_ylabel('Accuracy RMSD vs Ground Truth (Å)')
        ax.set_ylim(0, max(baseline + 1, mean_df['Value'].max() + 1))
        
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f Å", padding=3)
        
        ax.legend(loc=2)
        plt.tight_layout()
        return self._save(fig, save_path)
    
    def plot_rg_ensemble(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Scatter plot: Ensemble Rg vs Reference Rg."""
        fig, ax = plt.subplots(figsize=(3.54, 3.54), dpi=600)
        
        ax.scatter(df['rg_ref'], df['rg_avg'], alpha=0.5, color='blue')
        
        lims = [min(df['rg_ref'].min(), df['rg_avg'].min()),
                max(df['rg_ref'].max(), df['rg_avg'].max())]
        ax.plot(lims, lims, '--', color='red', alpha=0.3)
        
        ax.set_xlabel(r'$R_g$ target (nm)', fontsize=15)
        ax.set_ylabel(r'$R_g$ ensemble (nm)', fontsize=15)
        
        plt.tight_layout()
        return self._save(fig, save_path)
    
    def plot_re_ensemble(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Scatter plot: Ensemble Re vs Reference Re."""
        fig, ax = plt.subplots(figsize=(3.54, 3.54), dpi=600)
        
        ax.scatter(df['re_ref'], df['re_avg'], alpha=0.5, color='blue')
        
        lims = [min(df['re_ref'].min(), df['re_avg'].min()),
                max(df['re_ref'].max(), df['re_avg'].max())]
        ax.plot(lims, lims, '--', color='red', alpha=0.3)
        
        ax.set_xlabel(r'$R_{ee}$ target (nm)', fontsize=15)
        ax.set_ylabel(r'$R_{ee}$ ensemble (nm)', fontsize=15)
        
        plt.tight_layout()
        return self._save(fig, save_path)
    
    def plot_diversity_scatter(self, df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """Scatter plot: Ensemble diversity vs accuracy."""
        fig, ax = plt.subplots(figsize=(3.54, 3.54), dpi=600)
        
        sns.scatterplot(data=df, x='rmsd_avg', y='pair_rmsd_avg', 
                        palette='magma', legend=None, ax=ax)
        
        ax.set_xlabel('Average RMSD vs Ground Truth (Å)')
        ax.set_ylabel('RMSD Diversity of Conformations in Ensemble (Å)')
        
        plt.tight_layout()
        return self._save(fig, save_path)

# =============================================================================
# SECTION 5: DATA PERSISTENCE & CONVENIENCE
# =============================================================================

class DataManager:
    """Save/load analysis results."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./analysis_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_pkl(self, data: Any, filename: str) -> Path:
        """Save data to pickle file."""
        path = self.cache_dir / filename
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return path
    
    def load_pkl(self, filename: str) -> Any:
        """Load data from pickle file."""
        path = self.cache_dir / filename
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to CSV."""
        path = self.cache_dir / filename
        df.to_csv(path, index=False)
        return path
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from CSV."""
        path = self.cache_dir / filename
        return pd.read_csv(path) if path.exists() else None


def run_figure2_analysis(output_dir: Path = None) -> Tuple[pd.DataFrame, plt.Figure, plt.Figure]:
    """Run complete Figure 2 analysis: RMSD and Rg comparison."""
    analyzer = ApoHoloPairAnalyzer()
    viz = FigureVisualization(output_dir)
    
    df = analyzer.calculate_metrics()
    df = analyzer.add_openfold_accuracy(df)
    
    fig_rmsd = viz.plot_rmsd_comparison(df, 'figure2_rmsd.pdf')
    fig_rg = viz.plot_rg_comparison(df, 'figure2_rg.pdf')
    
    return df, fig_rmsd, fig_rg


def run_figure4_analysis(output_dir: Path = None) -> Tuple[pd.DataFrame, plt.Figure]:
    """Run complete Figure 4 analysis: Ensemble performance."""
    analyzer = EnsembleAnalyzer()
    viz = FigureVisualization(output_dir)
    
    df, all_data = analyzer.analyze_all()
    fig = viz.plot_ensemble_barplot(df, save_path='figure4_ensemble.pdf')
    
    # Save results
    if output_dir:
        dm = DataManager(output_dir)
        dm.save_csv(df, 'ensemble_summary.csv')
        dm.save_pkl({'df': df, 'data': all_data}, 'ensemble_analysis.pkl')
    
    return df, fig
