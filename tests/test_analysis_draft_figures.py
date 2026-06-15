"""
Tests for analysis_draft.py - Visualization
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class TestFigureVisualization:
    """Test FigureVisualization class."""
    
    @pytest.fixture
    def viz(self, tmp_path):
        """Create a FigureVisualization instance."""
        from metfish.analysis.analysis_draft import FigureVisualization
        return FigureVisualization(output_dir=tmp_path)
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'pdb_id': ['A', 'A', 'B', 'B', 'C', 'C'],
            'type': ['AlphaSAXS', 'OpenFold'] * 3,
            'rmsd': [2.0, 3.0, 1.5, 2.5, 4.0, 5.0],
            'rg_diff_A': [0.5, -0.3, 0.2, -0.1, 0.8, -0.5],
            'OpenFold Accuracy': ['High', 'High', 'Medium', 'Medium', 'Low', 'Low'],
        })
    
    def test_init_creates_output_dir(self, tmp_path):
        """Test that initialization creates output directory."""
        from metfish.analysis.analysis_draft import FigureVisualization
        
        output_dir = tmp_path / "test_output"
        viz = FigureVisualization(output_dir=output_dir)
        assert output_dir.exists()
    
    def test_plot_rmsd_comparison_returns_figure(self, viz, sample_df):
        """Test that plot_rmsd_comparison returns a Figure."""
        fig = viz.plot_rmsd_comparison(sample_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_rmsd_comparison_saves_file(self, viz, sample_df, tmp_path):
        """Test that plot_rmsd_comparison saves to file."""
        viz.plot_rmsd_comparison(sample_df, save_path='test_rmsd.pdf')
        assert (tmp_path / 'test_rmsd.pdf').exists()
        plt.close('all')
    
    def test_plot_rg_comparison_returns_figure(self, viz, sample_df):
        """Test that plot_rg_comparison returns a Figure."""
        fig = viz.plot_rg_comparison(sample_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_ensemble_barplot_returns_figure(self, viz):
        """Test that plot_ensemble_barplot returns a Figure."""
        df = pd.DataFrame({
            'rmsd_best': [2.0, 2.5, 3.0],
            'rmsd_avg': [3.0, 3.5, 4.0],
        })
        fig = viz.plot_ensemble_barplot(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_ensemble_barplot_with_baseline(self, viz):
        """Test ensemble barplot includes baseline."""
        df = pd.DataFrame({
            'rmsd_best': [2.0, 2.5, 3.0],
            'rmsd_avg': [3.0, 3.5, 4.0],
        })
        fig = viz.plot_ensemble_barplot(df, baseline=5.0)
        ax = fig.axes[0]
        # Check that there's a horizontal line (baseline)
        lines = [l for l in ax.get_lines() if len(l.get_xdata()) > 0]
        assert len(lines) > 0
        plt.close(fig)
    
    def test_plot_rg_ensemble_returns_figure(self, viz):
        """Test that plot_rg_ensemble returns a Figure."""
        df = pd.DataFrame({
            'rg_ref': [1.5, 2.0, 2.5],
            'rg_avg': [1.6, 2.1, 2.4],
        })
        fig = viz.plot_rg_ensemble(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_ensemble_rmsd_scatter_returns_figure(self, viz):
        """Test that plot_ensemble_rmsd_scatter returns a Figure."""
        apo_results = [{'rmsd': 2.0, 'rmsd_pair': 3.5} for _ in range(5)]
        holo_results = [{'rmsd': 2.5, 'rmsd_pair': 3.0} for _ in range(5)]
        
        fig = viz.plot_ensemble_rmsd_scatter(apo_results, holo_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotDendrogram:
    """Test plot_dendrogram function."""
    
    def test_plot_dendrogram_runs_without_error(self):
        """Test that plot_dendrogram executes without error."""
        from scipy.cluster.hierarchy import linkage
        from metfish.analysis.analysis_draft import plot_dendrogram
        
        # Create sample data and compute linkage
        data = np.random.randn(10, 5)
        linkage_matrix = linkage(data, method='ward')
        
        # This should not raise
        fig, ax = plt.subplots()
        plot_dendrogram(linkage_matrix)
        plt.close(fig)
