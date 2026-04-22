#!/usr/bin/env python3
"""
Generate SI Figures — All 80 Proteins (Apo-Holo Paired)
========================================================

Layout per page (US Letter, 8.5 x 11 in):
  - 4 protein rows  =  2 apo-holo pairs
  - Each row: [Ground Truth | OpenFold | AlphaSAXS | P(r) plot]
  - Apo & Holo rows are visually grouped with a pair header

Outputs a single multi-page PDF to  SI_figures/SI_all_proteins.pdf

Usage:
    python generate_si_figures.py                          # All 80 proteins
    python generate_si_figures.py --protein 1EEJ_B         # Single protein (debug)
    python generate_si_figures.py --pair 1TJD_A 1EEJ_B     # Single pair (debug)
    python generate_si_figures.py --rows-per-page 6        # Override layout
    python generate_si_figures.py --output-dir my_folder/
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB import PDBParser, Superimposer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DATA = Path("/global/cfs/cdirs/m4704/100125_Nature_Com_data")
CSV_PATH       = BASE_DATA / "results" / "model_comparisons.csv"
APO_HOLO_CSV   = BASE_DATA / "Apo_holo_data" / "Table_rmsd_Apo_vs_Holo.csv"
DIR_NMR = BASE_DATA / "single_conformation" / "nmr"       # AlphaSAXS
DIR_AF  = BASE_DATA / "single_conformation" / "af"        # OpenFold
DIR_REF = BASE_DATA / "Apo_holo_data" / "pdbs"            # Ground Truth

DEFAULT_OUTPUT = Path("SI_figures").resolve()

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
STRUCT_COLORS = {
    'Ground Truth': '#009E73',
    'OpenFold':     '#CC79A7',
    'AlphaSAXS':    '#E69F00',
}

RCPARAMS = {
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

# US Letter dimensions in inches
PAGE_W, PAGE_H = 8.5, 11.0
MARGIN_T = 0.55   # top margin (for page header)
MARGIN_B = 0.30   # bottom margin
PAIR_GAP = 0.18   # extra gap between apo-holo pairs (inches)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ca_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', str(pdb_path))
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                if 'CA' in residue:
                    coords.append(residue['CA'].get_vector().get_array())
        break
    return np.array(coords)


def _center_coords(coords):
    return coords - coords.mean(axis=0)


def _align_to_reference(ref_coords, mobile_coords):
    """Kabsch superposition of mobile onto reference. Returns (aligned, rmsd)."""
    n = min(len(ref_coords), len(mobile_coords))

    class _FA:
        def __init__(self, c):
            self._c = np.array(c, dtype='d')
        def get_coord(self):
            return self._c

    sup = Superimposer()
    sup.set_atoms([_FA(c) for c in ref_coords[:n]],
                  [_FA(c) for c in mobile_coords[:n]])
    rot, tran = sup.rotran
    return np.dot(mobile_coords, rot) + tran, sup.rms


def _compute_elev_azim(coords):
    """PCA-based viewing angle so the widest spread faces the viewer."""
    c = coords - coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    n = Vt[2]
    return (np.degrees(np.arcsin(np.clip(n[2], -1, 1))),
            np.degrees(np.arctan2(n[1], n[0])))


def _plot_structure_on_ax(ax, coords, color, title, rmsd=None,
                          elev=25, azim=-60):
    """Render aligned backbone trace on a 3D axis."""
    c = _center_coords(coords)
    # Thick backbone
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color, lw=2.0, alpha=0.92,
            solid_capstyle='round')
    # Shadow outline for depth
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color='k', lw=2.8, alpha=0.08,
            solid_capstyle='round')
    # CA beads
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=4, color=color, alpha=0.6,
               edgecolors='none', depthshade=True)
    # N/C terminus
    ax.scatter(*c[0],  s=25, color=color, marker='^', edgecolors='k',
               lw=0.4, zorder=5)
    ax.scatter(*c[-1], s=25, color=color, marker='s', edgecolors='k',
               lw=0.4, zorder=5)

    if rmsd is not None:
        ax.set_title('{}\nRMSD={:.2f} \u00c5'.format(title, rmsd),
                     fontsize=7, pad=1)
    else:
        ax.set_title(title, fontsize=7, pad=1)

    ax.view_init(elev=elev, azim=azim)
    rng = np.ptp(c, axis=0).max() / 2 * 1.15
    mid = c.mean(axis=0)
    for setter, idx in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        setter(mid[idx] - rng, mid[idx] + rng)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('w')
        axis.line.set_color('w')
        axis.set_ticks([])


def _clean_saxs(val):
    """Parse SAXS data from string or array."""
    if isinstance(val, str):
        val = val.strip()
        if ',' in val:
            return np.array(ast.literal_eval(val))
        return np.fromstring(
            val.replace('[', '').replace(']', '').replace('\n', ' '), sep=' ')
    return np.array(val)


def _get_pr_data(df, pid):
    """Return dict of (r, pr) arrays for AlphaSAXS, OpenFold, Ground Truth."""
    sn = df[(df['name'] == pid) & (df['comparison'] == 'out_NMR_vs_target')]
    sa = df[(df['name'] == pid) & (df['comparison'] == 'out_AF_vs_target')]
    if sn.empty or sa.empty:
        raise ValueError("Missing data for {}".format(pid))
    return {
        'AlphaSAXS':    (_clean_saxs(sn['saxs_bins_a'].iloc[0]),
                         _clean_saxs(sn['saxs_a'].iloc[0])),
        'OpenFold':     (_clean_saxs(sa['saxs_bins_a'].iloc[0]),
                         _clean_saxs(sa['saxs_a'].iloc[0])),
        'Ground Truth': (_clean_saxs(sn['saxs_bins_b'].iloc[0]),
                         _clean_saxs(sn['saxs_b'].iloc[0])),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_apo_holo_pairs(mc_df):
    """Return list of (apo_id, holo_id) for proteins present in mc_df."""
    ah = pd.read_csv(APO_HOLO_CSV, sep=';',
                     names=['Apo_ID', 'Holo_ID', 'RMSD'], skiprows=1)
    proteins = set(mc_df['name'].unique())
    seen, pairs = set(), []
    for _, row in ah.iterrows():
        a, h = row['Apo_ID'], row['Holo_ID']
        if a in proteins and h in proteins:
            key = tuple(sorted([a, h]))
            if key not in seen:
                seen.add(key)
                pairs.append((a, h))
    return pairs


def _prepare_protein(pid, mc_df):
    """Load, align, and return drawing data for one protein (or None).
    
    RMSD values are read from model_comparisons.csv (same source as Figure 2)
    rather than recomputed, ensuring consistency across all figures.
    """
    pdb_ref = DIR_REF / "{}.pdb".format(pid)
    pdb_af  = DIR_AF  / "{}.pdb".format(pid)
    pdb_nmr = DIR_NMR / "{}.pdb".format(pid)

    for p in (pdb_ref, pdb_af, pdb_nmr):
        if not p.exists():
            print("    SKIP {} - missing {}".format(pid, p.name))
            return None
    try:
        pr = _get_pr_data(mc_df, pid)
    except ValueError as e:
        print("    SKIP {} - {}".format(pid, e))
        return None

    # Look up RMSD from model_comparisons.csv
    af_row = mc_df[(mc_df['name'] == pid) & (mc_df['type_a'] == 'out_AF') & (mc_df['type_b'] == 'target')]
    nmr_row = mc_df[(mc_df['name'] == pid) & (mc_df['type_a'] == 'out_NMR') & (mc_df['type_b'] == 'target')]
    rmsd_af = af_row['rmsd'].iloc[0] if not af_row.empty else None
    rmsd_nmr = nmr_row['rmsd'].iloc[0] if not nmr_row.empty else None

    ref = _get_ca_coords(pdb_ref)
    af  = _get_ca_coords(pdb_af)
    nmr = _get_ca_coords(pdb_nmr)

    ref_c = _center_coords(ref)
    af_a,  _ = _align_to_reference(ref, af)
    nmr_a, _ = _align_to_reference(ref, nmr)
    elev, azim = _compute_elev_azim(ref_c)

    return dict(ref=ref, af=af_a, nmr=nmr_a,
                rmsd_af=rmsd_af, rmsd_nmr=rmsd_nmr,
                elev=elev, azim=azim, pr=pr, pid=pid)


# ---------------------------------------------------------------------------
# Draw one protein row
# ---------------------------------------------------------------------------

def _draw_protein_row(fig, data, row_bottom, row_height, is_apo=False):
    """Draw [GT struct | OF struct | AS struct | P(r)] at the given position.

    Coordinates are in figure-fraction [0,1].
    """
    pid = data['pid']

    # 4 equal columns with margins
    left_margin = 0.06
    right_margin = 0.02
    col_gap = 0.02
    usable_w = 1.0 - left_margin - right_margin
    col_w = (usable_w - 3 * col_gap) / 4
    lefts = [left_margin + i * (col_w + col_gap) for i in range(4)]

    # Vertical padding inside the row
    pad_b = row_height * 0.04
    pad_t = row_height * 0.06
    inner_h = row_height - pad_b - pad_t
    inner_b = row_bottom + pad_b

    # --- 3 structure panels ---
    structs = [
        (data['ref'], 'Ground Truth', STRUCT_COLORS['Ground Truth'], None),
        (data['af'],  'OpenFold',     STRUCT_COLORS['OpenFold'],     data['rmsd_af']),
        (data['nmr'], 'AlphaSAXS',    STRUCT_COLORS['AlphaSAXS'],   data['rmsd_nmr']),
    ]
    for i, (coords, label, color, rmsd) in enumerate(structs):
        ax = fig.add_axes([lefts[i], inner_b, col_w, inner_h], projection='3d')
        _plot_structure_on_ax(ax, coords, color, label, rmsd=rmsd,
                              elev=data['elev'], azim=data['azim'])

    # --- P(r) panel ---
    pr_b = inner_b + inner_h * 0.15
    pr_h = inner_h * 0.72
    ax_pr = fig.add_axes([lefts[3], pr_b, col_w, pr_h])
    for label in ['Ground Truth', 'OpenFold', 'AlphaSAXS']:
        r, pr = data['pr'][label]
        ax_pr.plot(r, pr, label=label, lw=1.0,
                   color=STRUCT_COLORS[label], alpha=0.85)
    ax_pr.set_xlabel('r (\u00c5)', labelpad=1)
    ax_pr.set_ylabel('P(r)', labelpad=1)
    ax_pr.set_title(pid, fontsize=7, pad=1)
    ax_pr.spines['top'].set_visible(False)
    ax_pr.spines['right'].set_visible(False)
    ax_pr.legend(frameon=False, fontsize=5, loc='upper right',
                 handlelength=1.2, handletextpad=0.4)
    if pid == '1EEJ_B':
        ax_pr.set_xlim(0, 70)
        ax_pr.set_ylim(0, 0.02)
    else:
        ax_pr.set_xlim(0, 100)
        ax_pr.set_ylim(0, 0.03)

    # --- Apo/Holo tag on left margin ---
    tag = 'Apo' if is_apo else 'Holo'
    fig.text(0.015, inner_b + inner_h / 2, tag, fontsize=7,
             rotation=90, va='center', ha='center',
             fontstyle='italic', color='#555555')


# ---------------------------------------------------------------------------
# Generate the full multi-page PDF
# ---------------------------------------------------------------------------

def generate_all_si(mc_df, output_dir, rows_per_page=4):
    plt.rcParams.update(RCPARAMS)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "SI_all_proteins.pdf"

    pairs = load_apo_holo_pairs(mc_df)
    print("  Found {} apo-holo pairs ({} proteins)".format(
        len(pairs), len(pairs) * 2))

    # Pre-load all data
    print("  Loading & aligning structures ...")
    pair_data = []
    for apo_id, holo_id in pairs:
        d_apo  = _prepare_protein(apo_id,  mc_df)
        d_holo = _prepare_protein(holo_id, mc_df)
        if d_apo and d_holo:
            pair_data.append((d_apo, d_holo))
        else:
            print("    Warning: Skipping pair {} / {}".format(apo_id, holo_id))
    print("  {} pairs ready".format(len(pair_data)))

    pairs_per_page = rows_per_page // 2
    page_groups = [pair_data[i:i + pairs_per_page]
                   for i in range(0, len(pair_data), pairs_per_page)]
    total_pages = len(page_groups)
    print("  {} pages ({} rows / page)\n".format(total_pages, rows_per_page))

    # Vertical layout fractions
    top_frac = 1.0 - MARGIN_T / PAGE_H
    bot_frac = MARGIN_B / PAGE_H
    usable_frac = top_frac - bot_frac

    with PdfPages(str(pdf_path)) as pdf:
        for pg_idx, page_pairs in enumerate(page_groups):
            n_rows = len(page_pairs) * 2
            n_pair_gaps = len(page_pairs) - 1
            gap_frac = PAIR_GAP / PAGE_H
            row_h = (usable_frac - n_pair_gaps * gap_frac) / n_rows

            fig = plt.figure(figsize=(PAGE_W, PAGE_H))

            # Page header
            fig.text(0.5, top_frac + 0.015,
                     'Supplementary Figure  -  Page {}/{}'.format(
                         pg_idx + 1, total_pages),
                     fontsize=10, ha='center', va='bottom', fontweight='bold')

            # Column headers (once per page, above first row)
            col_headers = ['Ground Truth', 'OpenFold', 'AlphaSAXS', 'P(r)']
            left_margin = 0.06
            col_gap = 0.02
            usable_w = 1.0 - left_margin - 0.02
            col_w = (usable_w - 3 * col_gap) / 4
            for ci, hdr in enumerate(col_headers):
                x = left_margin + ci * (col_w + col_gap) + col_w / 2
                fig.text(x, top_frac + 0.003, hdr, fontsize=7.5,
                         ha='center', va='bottom', color='#666666')

            cursor = top_frac
            for pair_i, (d_apo, d_holo) in enumerate(page_pairs):
                # Pair divider label
                pair_label = '--- {}  (Apo)    {}  (Holo) ---'.format(
                    d_apo['pid'], d_holo['pid'])
                fig.text(0.5, cursor - 0.003, pair_label,
                         fontsize=7.5, ha='center', va='top',
                         fontweight='bold', color='#333333')

                # Apo row
                cursor -= row_h
                _draw_protein_row(fig, d_apo, cursor, row_h, is_apo=True)

                # Holo row
                cursor -= row_h
                _draw_protein_row(fig, d_holo, cursor, row_h, is_apo=False)

                # Gap before next pair
                if pair_i < len(page_pairs) - 1:
                    sep_y = cursor - gap_frac / 2
                    fig.add_artist(plt.Line2D(
                        [0.06, 0.94], [sep_y, sep_y],
                        transform=fig.transFigure,
                        color='#cccccc', lw=0.5, ls='--'))
                cursor -= gap_frac

            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            names = [d['pid'] for p in page_pairs for d in p]
            print("  Page {:2d}/{}  {}".format(
                pg_idx + 1, total_pages, ', '.join(names)))

    size_mb = pdf_path.stat().st_size / 1e6
    print("\n  Saved {}  ({} pages, {:.1f} MB)".format(
        pdf_path, total_pages, size_mb))
    return pdf_path


# ---------------------------------------------------------------------------
# Single-protein / single-pair modes (for quick debugging)
# ---------------------------------------------------------------------------

def generate_single_si(pid, mc_df, output_dir):
    plt.rcParams.update(RCPARAMS)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_protein(pid, mc_df)
    if not data:
        return None
    fig = plt.figure(figsize=(PAGE_W, 2.5))
    _draw_protein_row(fig, data, 0.05, 0.85, is_apo=True)
    fig.suptitle('SI - {}'.format(pid), fontsize=10, y=0.98)
    pdf_path = output_dir / "SI_{}.pdf".format(pid)
    with PdfPages(str(pdf_path)) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  {} -> {}".format(pid, pdf_path))
    return pdf_path


def generate_pair_si(apo_id, holo_id, mc_df, output_dir):
    plt.rcParams.update(RCPARAMS)
    output_dir.mkdir(parents=True, exist_ok=True)
    d_apo  = _prepare_protein(apo_id,  mc_df)
    d_holo = _prepare_protein(holo_id, mc_df)
    if not (d_apo and d_holo):
        return None
    fig = plt.figure(figsize=(PAGE_W, PAGE_H / 2))
    _draw_protein_row(fig, d_apo,  0.52, 0.42, is_apo=True)
    _draw_protein_row(fig, d_holo, 0.05, 0.42, is_apo=False)
    fig.suptitle('{} (Apo)  /  {} (Holo)'.format(apo_id, holo_id),
                 fontsize=10, y=0.98, fontweight='bold')
    pdf_path = output_dir / "SI_pair_{}_{}.pdf".format(apo_id, holo_id)
    with PdfPages(str(pdf_path)) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Pair -> {}".format(pdf_path))
    return pdf_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SI figures (3 structures + P(r)) as multi-page PDF."
    )
    parser.add_argument('--protein', type=str, default=None,
                        help='Single protein ID for quick test')
    parser.add_argument('--pair', nargs=2, metavar=('APO', 'HOLO'),
                        help='Single apo-holo pair for quick test')
    parser.add_argument('--rows-per-page', type=int, default=4,
                        choices=[4, 6], help='Rows per page (default 4 = 2 pairs)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: {})'.format(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT

    print("Loading CSV: {}".format(CSV_PATH))
    mc_df = pd.read_csv(CSV_PATH)
    print("  {} proteins in CSV\n".format(mc_df['name'].nunique()))

    if args.protein:
        generate_single_si(args.protein, mc_df, output_dir)
    elif args.pair:
        generate_pair_si(args.pair[0], args.pair[1], mc_df, output_dir)
    else:
        generate_all_si(mc_df, output_dir, rows_per_page=args.rows_per_page)


if __name__ == '__main__':
    main()
