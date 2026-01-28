"""
Analysis module for metfish.

Contains:
- processor.py: ModelComparisonProcessor, ProteinStructureAnalyzer
- visualizer.py: ProteinVisualization
- analysis_draft.py: Extended analysis functions (draft for integration)
"""

from metfish.analysis.processor import (
    ProteinStructureAnalyzer,
    ModelComparisonProcessor,
)
from metfish.analysis.visualizer import ProteinVisualization

# Draft module exports (for testing - will be merged later)
from metfish.analysis.analysis_draft import (
    # Shared utilities
    DataPaths,
    analyze_structure,
    calc_ca_rmsd,
    calc_pr_l1_loss,
    # Figure 2 & 3
    ApoHoloPairAnalyzer,
    # Figure 4
    EnsembleAnalyzer,
    ContactMapAnalyzer,
    plot_dendrogram,
    calc_pairwise_rmsd_optimized,
    calc_ensemble_diversity,
    # Visualization
    FigureVisualization,
    # Data persistence
    DataManager,
    # Convenience functions
    run_figure2_analysis,
    run_figure4_analysis,
)

__all__ = [
    # Original modules
    'ProteinStructureAnalyzer',
    'ModelComparisonProcessor',
    'ProteinVisualization',
    # Draft module - shared
    'DataPaths',
    'analyze_structure',
    'calc_ca_rmsd',
    'calc_pr_l1_loss',
    # Draft module - analyzers
    'ApoHoloPairAnalyzer',
    'EnsembleAnalyzer',
    'ContactMapAnalyzer',
    'plot_dendrogram',
    # Draft module - visualization
    'FigureVisualization',
    # Draft module - data
    'DataManager',
    'run_figure2_analysis',
    'run_figure4_analysis',
]
