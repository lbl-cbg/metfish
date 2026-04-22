"""Generate a Nature Communications-style cover page for the Supplementary Information PDF."""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def generate_cover_page(output_path="supplementary_cover_page.pdf"):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Top line accent (Nature Communications style)
    ax.axhline(y=0.88, xmin=0.08, xmax=0.92, color='#C8102E', linewidth=2.5)

    # "Supplementary Information" header
    ax.text(
        0.5, 0.84,
        "Supplementary Information",
        fontsize=24, fontweight='bold', color='#333333',
        ha='center', va='center',
    )

    # Thin separator
    ax.axhline(y=0.80, xmin=0.30, xmax=0.70, color='#AAAAAA', linewidth=0.8)

    # Main title
    title = (
        "Experimental Data-Driven AI Framework\n"
        "for Flexible Protein Conformational Reconstruction"
    )
    ax.text(
        0.5, 0.70,
        title,
        fontsize=18, fontweight='bold', color='#222222',
        ha='center', va='center', linespacing=1.5,
        wrap=False,
    )

    # Subtitle
    subtitle = (
        "AlphaSAXS Predicted Structures vs. OpenFold Predictions\n"
        "and Ground Truth PDB Structures"
    )
    ax.text(
        0.5, 0.58,
        subtitle,
        fontsize=14, fontstyle='italic', color='#555555',
        ha='center', va='center', linespacing=1.5,
    )

    # Bottom separator
    ax.axhline(y=0.50, xmin=0.20, xmax=0.80, color='#AAAAAA', linewidth=0.5)

    # Description block
    description = (
        "This supplementary document presents per-protein structural comparisons\n"
        "across 40 apo-holo pairs within the test dataset. Each page shows the\n"
        "ground-truth PDB structure alongside predictions from OpenFold and\n"
        "AlphaSAXS, with corresponding pair-distance distribution functions P(r)."
    )
    ax.text(
        0.5, 0.42,
        description,
        fontsize=11, color='#444444',
        ha='center', va='center', linespacing=1.6,
    )

    # Bottom accent line
    ax.axhline(y=0.10, xmin=0.08, xmax=0.92, color='#C8102E', linewidth=2.5)

    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.05)

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)

    plt.close(fig)
    print(f"Cover page saved to: {output_path}")


if __name__ == "__main__":
    generate_cover_page()
