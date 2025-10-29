import pandas as pd
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
from typing import List, Dict, Optional


class ProteinVisualization:
    """Visualization tools for protein structure comparison."""
    
    def __init__(self, color_scheme: Dict[str, str], label_dict: Dict[str, str]):
        self.color_scheme = color_scheme
        self.label_dict = label_dict
        
    def map_labels(self, text: str) -> str:
        """Map internal labels to display labels."""
        for k, v in self.label_dict.items():
            text = text.replace(k, v)
        return text
    
    def plot_overall_metrics(self,
                            comparison_df: pd.DataFrame,
                            models: List[str],
                            metrics: List[str],
                            save_path: Optional[Path] = None):
        """Create overall model performance comparison plots."""
        # Prepare data
        comparisons = [f"out_{m}_vs_target" for m in models]
        comparison_labels = [self.map_labels(c) for c in comparisons]
        
        group_df = (comparison_df[['name', 'comparison', *metrics]]
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
                       palette=[self.color_scheme[c] for c in comparison_labels],
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
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        return fig, stats_df
    
    def plot_model_improvement(self,
                              comparison_df: pd.DataFrame,
                              base_model: str,
                              comparison_models: List[str],
                              metrics: List[str] = ['rmsd', 'saxs_kldiv'],
                              save_path: Optional[Path] = None):
        """Plot improvement of models compared to baseline."""
        fig, axes = plt.subplots(1, len(metrics), figsize=(10, 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, (ax, metric) in enumerate(zip(axes, metrics)):
            # Pivot data for comparison
            metric_df = self._prepare_improvement_data(
                comparison_df, base_model, comparison_models, metric
            )
            
            # Scatter plot
            for model in comparison_models:
                col_name = self.map_labels(f'out_{model}_vs_target')
                sns.scatterplot(data=metric_df,
                               x=f'{base_model} vs. Target',
                               y=col_name,
                               ax=ax,
                               color=self.color_scheme[col_name],
                               label=col_name.split(" vs. Target")[0])
            
            # Identity line
            identity_line = [ax.get_xlim()[0], ax.get_xlim()[-1]]
            ax.plot(identity_line, identity_line, '--k', alpha=0.75, zorder=0)
            ax.set(title=metric, xlabel=f'{base_model} vs. Target', ylabel='Model vs. Target')
            
            # Add arrow annotation
            ax.annotate('', xy=(0.5, 0.25), xytext=(0.7, 0.25),
                       xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='<|-'))
            ax.annotate('better prediction', xy=(0.75, 0.25),
                       xycoords='axes fraction', ha='left', va='center', fontsize=9)
        
        plt.suptitle(f'Model Performance vs. {base_model}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        return fig
    
    def _prepare_improvement_data(self,
                                 comparison_df: pd.DataFrame,
                                 base_model: str,
                                 comparison_models: List[str],
                                 metric: str) -> pd.DataFrame:
        """Prepare data for improvement visualization."""
        comparisons = [f"out_{m}_vs_target" for m in [base_model] + comparison_models]
        comparison_labels = [self.map_labels(c) for c in comparisons]
        
        group_df = (comparison_df[['name', 'comparison', metric]]
                   .query(f'comparison in {comparisons}')
                   .assign(labels_comparison=lambda x: x['comparison'].apply(self.map_labels)))
        
        metric_df = (group_df
                    .pivot(columns='labels_comparison', index='name', values=metric)
                    .reset_index())
        
        return metric_df
    
    def create_structure_viewer(self,
                               fname_a: str,
                               fname_b: str,
                               label_a: str,
                               label_b: str,
                               width: int = 400,
                               height: int = 400) -> py3Dmol.view:
        """Create 3D structure visualization."""
        view = py3Dmol.view(width=width, height=height)
        
        # Load structures
        view.addModel(open(fname_a, 'r').read(), 'pdb')
        view.addModel(open(fname_b, 'r').read(), 'pdb')
        
        # Style structures
        view.setStyle({'model': 0}, 
                     {"cartoon": {'color': self.color_scheme.get(label_a, 'gray')}})
        view.setStyle({'model': 1}, 
                     {"cartoon": {'color': self.color_scheme.get(label_b, 'blue')}})
        
        # Add labels
        view.addLabel(label_a,
                     {'position': {'x': 0, 'y': 10, 'z': 0},
                      'backgroundColor': 'white',
                      'fontColor': self.color_scheme.get(label_a, 'gray'),
                      'fontSize': 14})
        view.addLabel(label_b,
                     {'position': {'x': 0, 'y': 20, 'z': 0},
                      'backgroundColor': 'white',
                      'fontColor': self.color_scheme.get(label_b, 'blue'),
                      'fontSize': 14})
        
        view.zoomTo()
        return view