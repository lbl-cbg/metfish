import pandas as pd
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional


class ProteinVisualization:
    """Visualization tools for protein structure comparison."""
    
    def __init__(self, df: pd.DataFrame, color_scheme: Dict[str, str], output_dir: Optional[Path] = None):
        self.df = df
        self.color_scheme = color_scheme
        self.models = ['AF', 'NMR', 'NMA']
        self.output_dir = output_dir
        
        # Create output directory if specified
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def map_labels(self, text: str) -> str:
        """Map internal labels to display labels."""
        return text.replace('_vs_', ' vs. ').replace('_', ' ').replace('out ', '')
    
    def get_palette(self, labels: List[str]) -> str:
        """Select color based on label."""
        palette = []
        for label in labels:
            color = next((v for k, v in self.color_scheme.items() if k in label))
            palette.append(color)

        return palette
    
    def plot_all(self):
        self.plot_overall_metrics(['rmsd', 'lddt', 'saxs_kldiv'], 'overall_metrics.pdf')
        self.plot_model_improvement('model_improvement.pdf')
        self.plot_protein_summary_all()
    
    def plot_overall_metrics(self,
                            metrics: List[str],
                            save_path: Path):
        """Create overall model performance comparison plots."""
        # Prepare data
        comparisons = [f"out_{m.replace('SFold_', '')}_vs_target" for m in self.models]
        comparison_labels = [self.map_labels(c) for c in comparisons]
        
        group_df = (self.df[['name', 'comparison', *metrics]]
                   .query(f'comparison in {comparisons}')
                   .melt(id_vars=['name', 'comparison'], 
                        value_vars=metrics, 
                        var_name='metric', 
                        value_name='value')
                   .assign(labels_comparison=lambda x: x['comparison'].apply(self.map_labels)))
        
        # Calculate statistics
        stats_df = (group_df
                   .groupby(['metric', 'labels_comparison'])['value']
                   .agg(['median', 'mean', 'std', 'count'])
                   .assign(se=lambda x: x['std']/np.sqrt(x['count'])))
        
        # Create plots
        fig, axes = plt.subplots(1, len(metrics), figsize=(7*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
            
        for i, (ax, metric) in enumerate(zip(axes, metrics)):
            metric_data = group_df.query(f'metric == "{metric}"')
            
            # Box plot
            sns.boxplot(ax=ax,
                       data=metric_data, 
                       x='labels_comparison', 
                       y='value', 
                       order=comparison_labels, 
                       hue='labels_comparison',
                       palette=self.get_palette(comparison_labels),
                       showfliers=False, width=0.25, boxprops=dict(alpha=.5))
            
            # Individual points
            sns.pointplot(ax=ax,
                         data=metric_data,
                         x='labels_comparison',
                         y='value',
                         hue='name',
                         order=comparison_labels,
                         palette='dark:k',
                         alpha=0.1,
                         markersize=7,
                         linewidth=0.75,
                         legend=False)
            
            # Add statistics labels
            for j, comparison in enumerate(comparison_labels):
                median = stats_df.loc[(metric, comparison), 'median']
                se = stats_df.loc[(metric, comparison), 'se']
                ax.text(j, ax.get_ylim()[0]*1.1, 
                       f'{median:.4f}±{se:.2f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set(title=f'{metric}', xlabel=None, ylabel=metric)
        
        plt.suptitle("Model Performance Comparison")
        plt.tight_layout()
        
        
        save_path = self.output_dir / save_path if self.output_dir is not None else save_path
        plt.savefig(save_path, dpi=300)
        
        return fig, stats_df
    
    def plot_model_improvement(self,
                              save_path: Path,
                              metrics: List[str] = ['rmsd', 'saxs_kldiv', 'rg_diff'],
                              ):
        """Plot improvement of models compared to baseline."""
        fig, axes = plt.subplots(1, len(metrics), figsize=(10, 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, (ax, metric) in enumerate(zip(axes, metrics)):
            # Pivot data for comparison
            metric_df = self._prepare_improvement_data(self.df, metric)
            
            # Scatter plot
            for model in ['NMR', 'NMA']:
                col_name = self.map_labels(f'out_{model}_vs_target')
                sns.scatterplot(data=metric_df,
                               x='AF vs. target',
                               y=col_name,
                               ax=ax,
                               color=self.get_palette([col_name])[0],
                               label=col_name.split(" vs. target")[0])
            
            # Identity line
            identity_line = [ax.get_xlim()[0], ax.get_xlim()[-1]]
            ax.plot(identity_line, identity_line, '--k', alpha=0.75, zorder=0)
            ax.set(title=metric, xlabel='AF vs. target', ylabel='Model vs. target')
            
            # Add arrow annotation
            ax.annotate('', xy=(0.5, 0.25), xytext=(0.7, 0.25),
                       xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='<|-'))
            ax.annotate('better prediction', xy=(0.75, 0.25),
                       xycoords='axes fraction', ha='left', va='center', fontsize=9)
        
        plt.suptitle('Model Performance vs. AlphaFold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_path if self.output_dir is not None else save_path
        plt.savefig(save_path, dpi=300)
        
        return fig
    
    def _prepare_improvement_data(self,
                                 comparison_df: pd.DataFrame,
                                 metric: str) -> pd.DataFrame:
        """Prepare data for improvement visualization."""
        comparisons = [f"out_{m}_vs_target" for m in self.models]
        
        group_df = (comparison_df[['name', 'comparison', metric]]
                   .query(f'comparison in {comparisons}')
                   .assign(labels_comparison=lambda x: x['comparison'].apply(self.map_labels))
                   .pivot(columns='labels_comparison', index='name', values=metric)
                   .reset_index())
        
        return group_df
    
    def plot_protein_summary_all(self, 
                                 proteins_per_page: int = 20,
                                 save_path_prefix: str = 'protein_summary_grid',
                                 ncols: int = 4,
                                 figsize: tuple = (20, 25)):
        """
        Plot summaries for all proteins in the dataframe, creating multiple PDF files
        with a specified number of proteins per page.
        """
        # Get all unique protein names
        all_protein_names = self.df['name'].unique()
        
        # Create figures for each page
        n_pages = int(np.ceil(len(all_protein_names) / proteins_per_page))
        for page_num in range(n_pages):
            start_idx = page_num * proteins_per_page
            end_idx = min(start_idx + proteins_per_page, len(all_protein_names))
            protein_names = all_protein_names[start_idx:end_idx]
            
            # Create figure with subplots
            nrows = int(np.ceil(proteins_per_page / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = axes.flatten() 
            
            # Plot each protein
            for idx, name in enumerate(protein_names):
                self._plot_protein_saxs_summary(name, axes[idx])

            fig.suptitle(f'Protein Summary Grid - {page_num + 1}/{n_pages} ')

            save_path = f'{save_path_prefix}_page{page_num + 1}.pdf'            
            save_path = self.output_dir / save_path if self.output_dir is not None else save_path
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
                    
    def plot_protein_structures_all(self):
        """Plot comparisons for all proteins in the dataframe."""
        for name in self.df['name'].unique():
            self.plot_protein_structures(name, save_path=f'protein_{name}_structures.pdf')

    def _plot_protein_saxs_summary(self, name: str, ax: plt.Axes):
        """
        Create a compact visualization for a single protein in a subplot.
        Shows SAXS curves with RMSD values as text annotations.
        """
        # Filter data for the specified protein and comparisons
        labels = [m.replace('SFold_', '') for m in self.models]
        comparisons = [f"out_{label}_vs_target" for label in labels]

        comparison_labels = [self.map_labels(c) for c in comparisons]    
        data = self.df.query('comparison in @comparisons & name == @name').copy()

        # Prepare data for plotting
        for col in ['type_a', 'type_b', 'comparison']:
            data[f'labels_{col}'] = data[col].apply(self.map_labels)

        subset_df = (pd.concat([
            data[['labels_type_a', 'saxs_bins_a', 'saxs_a', 'rg_a', 'fname_a']].rename(
                columns=lambda x: x.rstrip('_a')),
            data[['labels_type_b', 'saxs_bins_b', 'saxs_b', 'rg_b', 'fname_b']].rename(
                columns=lambda x: x.rstrip('_b'))
        ]).drop_duplicates('labels_type'))
        saxs_data = subset_df[['labels_type', 'saxs_bins', 'saxs', 'rg']].explode(
            ['saxs_bins', 'saxs'])
        
        # Get color palettes
        palette = self.get_palette(comparison_labels)
        
        # Plot SAXS data
        sns.lineplot(data=saxs_data, x='saxs_bins', y='saxs',
                    hue='labels_type', hue_order=[*labels, "target"], 
                    palette=[*palette, self.get_palette(["Target"])[0]], 
                    ax=ax, linewidth=1.5, legend=True)
        ax.legend(fontsize=8, loc='upper right', frameon=True, title='Model')

        # Prepare RMSD data for text annotation
        rmsd_data = data[['labels_comparison', 'rmsd']].drop_duplicates()
        rmsd_data = rmsd_data.sort_values('labels_comparison')
        
        # Add RMSD values as text annotation in upper right corner
        rmsd_text = "RMSD (Å) vs. target:\n"
        for _, row in rmsd_data.iterrows():
            comparison = row['labels_comparison'].replace(' vs. target', '')
            rmsd_text += f"{comparison}: {row['rmsd']:.2f}\n"
        
        ax.text(0.98, 0.5, rmsd_text.rstrip('\n'), 
               transform=ax.transAxes, 
               fontsize=8, 
               verticalalignment='top',
               horizontalalignment='right',)
        
        # Formatting
        ax.set_xlabel('r (Å)', fontsize=8)
        ax.set_ylabel('P(r)', fontsize=8)
        ax.set_title(name, fontsize=9, fontweight='bold')
        sns.despine(ax=ax)
    
