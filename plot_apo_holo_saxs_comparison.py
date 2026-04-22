#!/usr/bin/env python3
"""
Generate separate SAXS P(r) comparison figures for apo and holo proteins.
- Apo prediction vs Apo ground truth
- Holo prediction vs Holo ground truth
- Metrics printed as a table to stdout
- Ground truth in blue, legend placed outside the plot area
Pairs:
  Pair 1: Apo=1JFJ-3_A  Holo=1JFK_A
  Pair 2: Apo=1TJD_A    Holo=1EEJ_B
"""

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from pathlib import Path
from metfish.utils import get_Pr
from metfish.analysis.analysis_draft import calc_pr_l1_loss, calc_ca_rmsd
from metfish.analysis.plot_config import apply_plot_style, FIGURE_SIZES, COLORS

# Paths
BASE_DATA = Path("/global/cfs/cdirs/m4704/100125_Nature_Com_data")
REF_PATH = BASE_DATA / "Apo_holo_data" / "pdbs"
ALPHASAXS_PATH = BASE_DATA / "results" / "NMR"
OPENFOLD_PATH = BASE_DATA / "results" / "AF"
OUTPUT_DIR = Path("analysis_output_test")

# Both apo-holo pairs: (apo_id, holo_id)
PAIRS = [
    ("1JFJ-3_A", "1JFK_A"),
    ("1TJD_A",   "1EEJ_B"),
]

table_rows = []


def load_profiles(pred_id):
    """Load P(r) profiles for one protein. Returns (r_ref, pr_ref, r_as, pr_as) or None."""
    ref_pdb       = REF_PATH       / f"{pred_id}.pdb"
    alphasaxs_pdb = ALPHASAXS_PATH / f"{pred_id}_SFold_NMR.pdb"
    openfold_pdb  = OPENFOLD_PATH  / f"{pred_id}_AlphaFold_AF.pdb"

    for path, name in [(ref_pdb, "Reference"), (alphasaxs_pdb, "AlphaSAXS"), (openfold_pdb, "OpenFold")]:
        if not path.exists():
            print(f"  Error: {name} PDB not found: {path}")
            return None

    r_ref, pr_ref = get_Pr(str(ref_pdb),      None, None, 0.5)
    r_as,  pr_as  = get_Pr(str(alphasaxs_pdb), None, None, 0.5)
    return r_ref, pr_ref, r_as, pr_as, ref_pdb, alphasaxs_pdb, openfold_pdb


def plot_saxs_comparison(pred_id, label, output_name, xlim, ylim, profiles):
    """Plot with pre-computed uniform axis limits."""
    r_ref, pr_ref, r_as, pr_as, ref_pdb, alphasaxs_pdb, openfold_pdb = profiles

    # Metrics
    l1_as = calc_pr_l1_loss(str(ref_pdb), str(alphasaxs_pdb), bin_size=0.5)
    l1_of = calc_pr_l1_loss(str(ref_pdb), str(openfold_pdb),  bin_size=0.5)
    l1_impr = (l1_of - l1_as) / l1_of * 100 if l1_of > 0 else 0.0

    traj_ref = md.load(str(ref_pdb))
    traj_as  = md.load(str(alphasaxs_pdb))
    traj_of  = md.load(str(openfold_pdb))
    rmsd_as  = calc_ca_rmsd(traj_as, traj_ref) or 0.0
    rmsd_of  = calc_ca_rmsd(traj_of, traj_ref) or 0.0
    rmsd_impr = (rmsd_of - rmsd_as) / rmsd_of * 100 if rmsd_of > 0 else 0.0

    table_rows.append((pred_id, label, l1_as, l1_of, l1_impr, rmsd_as, rmsd_of, rmsd_impr))

    pred_color = COLORS['apo'] if label == 'Apo' else COLORS['holo']

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])

    ax.plot(r_ref, pr_ref, color='blue',     label='Ground Truth')
    ax.plot(r_as,  pr_as,  color=pred_color, label='AlphaSAXS')

    ax.set_xlabel('r (Å)')
    ax.set_ylabel('P(r)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend(loc='upper left', bbox_to_anchor=(0.55, 1.02), borderaxespad=0)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def print_table():
    col_w = [12, 5, 10, 10, 12, 10, 10, 12]
    header = ["Protein", "Type", "AS L1", "OF L1", "L1 Impr%", "AS RMSD", "OF RMSD", "RMSD Impr%"]
    sep = "  ".join("-" * w for w in col_w)
    fmt_h = "  ".join(f"{h:<{w}}" for h, w in zip(header, col_w))
    print(f"\n{'='*len(sep)}")
    print(fmt_h)
    print(sep)
    for pred_id, label, l1_as, l1_of, l1_impr, rmsd_as, rmsd_of, rmsd_impr in table_rows:
        l1_sym   = "↓" if l1_impr   >= 0 else "↑"
        rmsd_sym = "↓" if rmsd_impr >= 0 else "↑"
        row = [
            f"{pred_id:<12}", f"{label:<5}",
            f"{l1_as:<10.4f}", f"{l1_of:<10.4f}", f"{l1_sym}{abs(l1_impr):<11.1f}%",
            f"{rmsd_as:<10.2f}", f"{rmsd_of:<10.2f}", f"{rmsd_sym}{abs(rmsd_impr):<11.1f}%",
        ]
        print("  ".join(row))
    print(f"{'='*len(sep)}\n")


def main():
    apply_plot_style()
    print("Generating SAXS P(r) comparison figures...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: load all profiles and compute global axis limits ---
    jobs = []
    for apo_id, holo_id in PAIRS:
        for pid, label in [(apo_id, "Apo"), (holo_id, "Holo")]:
            fname = f"saxs_{'apo' if label == 'Apo' else 'holo'}_{pid}_vs_gt.png"
            profiles = load_profiles(pid)
            if profiles is None:
                continue
            jobs.append((pid, label, fname, profiles))

    all_r  = np.concatenate([np.concatenate([p[0], p[2]]) for *_, p in jobs])
    all_pr = np.concatenate([np.concatenate([p[1], p[3]]) for *_, p in jobs])
    xlim = (0, float(np.max(all_r))  * 1.02)
    ylim = (0, float(np.max(all_pr)) * 1.08)
    print(f"\nUniform axes  x: [0, {xlim[1]:.1f}]  y: [0, {ylim[1]:.5f}]")

    # --- Pass 2: plot all four with uniform limits ---
    for pid, label, fname, profiles in jobs:
        print(f"\n  {label}: {pid}")
        plot_saxs_comparison(pid, label, fname, xlim, ylim, profiles)

    print_table()
    print("Done!")


if __name__ == "__main__":
    main()

