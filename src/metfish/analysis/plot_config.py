"""
Publication-ready plot configuration for metfish analysis figures.

This module defines standard rcParams for all figures to ensure consistency
across the manuscript. Import and apply these settings at the start of any
plotting script.

Usage:
    from metfish.analysis.plot_config import apply_plot_style, COLORS
    
    apply_plot_style()
    plt.plot(x, y, color=COLORS['alphaSAXS'])
"""

import matplotlib.pyplot as plt


# =============================================================================
# Standard Figure Sizes (inches)
# =============================================================================
FIGURE_SIZES = {
    'single': (3.35, 3.35),      # Single square plot
    'wide': (7.08, 4),            # Wide plot for full two-column width
}


# =============================================================================
# Color Palette
# =============================================================================
COLORS = {
    # Model colors
    'alphaSAXS': '#7B68EE',      # Purple - AlphaSAXS predictions
    'openfold': '#48A9A6',       # Teal - OpenFold predictions  
    'ground_truth': '#2ecc71',   # Green - Reference/ground truth
    
    # Apo-holo colors
    'apo': 'orange',             # Orange - Apo state
    'holo': 'green',             # Green - Holo state
    
    # Three-model comparison
    'model_blue': 'blue',        # Blue - OpenFold in three-model plots
    'model_orange': 'orange',    # Orange - AlphaSAXS in three-model plots
    'model_green': 'green',      # Green - Ground truth in three-model plots
    
    # Additional colors
    'baseline': '#e74c3c',       # Red - Baseline reference lines
    'identity': '#878787',       # Gray - Identity/diagonal lines
}


# =============================================================================
# Standard rcParams for Publication-Quality Figures
# =============================================================================
PLOT_PARAMS = {
    # Font settings
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    
    # Font family
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    
    # PDF settings (for editable text in PDFs)
    'pdf.fonttype': 42,
    
    # Figure settings
    'figure.dpi': 100,           # Screen display DPI
    'savefig.dpi': 600,          # Save DPI for publication
    'savefig.bbox': 'tight',     # Tight bounding box
    
    # Line and marker settings
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    
    # Axes settings
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelpad': 4,
    'axes.titlepad': 4,
    
    # Grid settings
    'axes.grid': False,
    
    # Legend settings
    'legend.frameon': False,
    'legend.loc': 'best',
    
    # Tick settings
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
}


# =============================================================================
# Style Application Functions
# =============================================================================

def apply_plot_style():
    """
    Apply standard plot style for all metfish analysis figures.
    
    Call this function at the start of any plotting script to ensure
    consistent formatting across all figures.
    """
    plt.rcParams.update(PLOT_PARAMS)


def reset_plot_style():
    """Reset matplotlib rcParams to defaults."""
    plt.rcParams.update(plt.rcParamsDefault)


def get_figure_size(size_name='single'):
    """
    Get standard figure size.
    
    Parameters
    ----------
    size_name : str
        Name of figure size: 'single' or 'wide'
        
    Returns
    -------
    tuple
        (width, height) in inches
    """
    return FIGURE_SIZES.get(size_name, FIGURE_SIZES['single'])


def setup_axes(ax, xlabel=None, ylabel=None, title=None, 
               remove_top_right=True):
    """
    Apply standard formatting to matplotlib axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Axes title
    remove_top_right : bool, default=True
        Whether to remove top and right spines
    """
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    if title:
        ax.set_title(title, pad=4)
    
    if remove_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# =============================================================================
# Specialized Color Palettes
# =============================================================================

def get_model_palette():
    """
    Get color palette for model comparison plots.
    
    Returns
    -------
    dict
        Dictionary mapping model names to colors
    """
    return {
        'AlphaSAXS': COLORS['alphaSAXS'],
        'OpenFold': COLORS['openfold'],
        'NMR': COLORS['alphaSAXS'],  # Same as AlphaSAXS
    }


def get_three_model_colors():
    """
    Get colors for three-model comparison plots.
    
    Returns
    -------
    dict
        Dictionary with 'openfold', 'alphaSAXS', and 'ground_truth' keys
    """
    return {
        'openfold': COLORS['model_blue'],
        'alphaSAXS': COLORS['model_orange'],
        'ground_truth': COLORS['model_green'],
    }


def get_apo_holo_colors():
    """
    Get colors for apo-holo comparison plots.
    
    Returns
    -------
    dict
        Dictionary with 'apo' and 'holo' keys
    """
    return {
        'apo': COLORS['apo'],
        'holo': COLORS['holo'],
    }
