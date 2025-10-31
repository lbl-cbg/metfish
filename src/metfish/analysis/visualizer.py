import pandas as pd
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union
import tempfile


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
        self.plot_pymol_protein_comparison('example_protein', save_path='pymol_example_protein.pdf')
    
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
                axes[idx] = self._plot_protein_summary_compact(name, axes[idx])

            fig.suptitle(f'Protein Summary Grid - {page_num + 1}/{n_pages} ')

            save_path = f'{save_path_prefix}_page{page_num + 1}.pdf'            
            save_path = self.output_dir / save_path if self.output_dir is not None else save_path
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
                    
    def plot_protein_structures_all(self):
        """Plot comparisons for all proteins in the dataframe."""
        for name in self.df['name'].unique():
            self.plot_protein_structures(name, save_path=f'protein_{name}_structures.pdf')

    def _plot_protein_summary_compact(self, name: str, ax: plt.Axes):
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
                    ax=ax, linewidth=1.5, legend=False)
        
        # Prepare RMSD data for text annotation
        rmsd_data = data[['labels_comparison', 'rmsd']].drop_duplicates()
        rmsd_data = rmsd_data.sort_values('labels_comparison')
        
        # Add RMSD values as text annotation in upper right corner
        rmsd_text = "RMSD (Å) vs. target:\n"
        for _, row in rmsd_data.iterrows():
            comparison = row['labels_comparison'].replace(' vs. target', '')
            rmsd_text += f"{comparison}: {row['rmsd']:.2f}\n"
        
        ax.text(0.98, 0.98, rmsd_text.rstrip('\n'), 
               transform=ax.transAxes, 
               fontsize=6, 
               verticalalignment='top',
               horizontalalignment='right',)
        
        # Formatting
        ax.set_xlabel('r (Å)', fontsize=8)
        ax.set_ylabel('P(r)', fontsize=8)
        ax.set_title(name, fontsize=9, fontweight='bold')

        return ax
    
    # def plot_protein_summary(self,
    #                         name: str,
    #                         save_path: Path,
    #                         figsize: tuple = (15, 6)) -> plt.Figure:
    #     """
    #     Create a comprehensive visualization for a single protein showing SAXS data
    #     with Rg annotations and an RMSD comparison table.
    #     """
    #     # Filter data for the specified protein and comparisons
    #     labels = [m.replace('SFold_', '') for m in self.models]
    #     comparisons = [f"out_{label}_vs_target" for label in labels]

    #     comparison_labels = [self.map_labels(c) for c in comparisons]    
    #     data = self.df.query('comparison in @comparisons & name == @name').copy()

    #     # Prepare data for plotting
    #     for col in ['type_a', 'type_b', 'comparison']:
    #         data[f'labels_{col}'] = data[col].apply(self.map_labels)

    #     subset_df = (pd.concat([
    #         data[['labels_type_a', 'saxs_bins_a', 'saxs_a', 'rg_a', 'fname_a']].rename(
    #             columns=lambda x: x.rstrip('_a')),
    #         data[['labels_type_b', 'saxs_bins_b', 'saxs_b', 'rg_b', 'fname_b']].rename(
    #             columns=lambda x: x.rstrip('_b'))
    #     ]).drop_duplicates('labels_type'))
    #     saxs_data = subset_df[['labels_type', 'saxs_bins', 'saxs', 'rg']].explode(
    #         ['saxs_bins', 'saxs'])
        
    #     # Get color palettes
    #     palette = self.get_palette(comparison_labels)
        
    #     # Create figure 
    #     fig, ax = plt.subplots(1, 2, figsize=figsize, width_ratios=[3, 1])
        
    #     # Plot SAXS data (left panel)
    #     sns.lineplot(data=saxs_data, x='saxs_bins', y='saxs',
    #                 hue='labels_type', hue_order=[*labels, "target"], 
    #                 palette=[*palette, self.get_palette(["Target"])[0]], ax=ax[0], linewidth=2)
        
    #     ax[0].set(title='SAXS data', xlabel='r', ylabel='P(r)')
            
    #     # Prepare RMSD data for table
    #     ax[1].axis('off')
    #     rmsd_data = data[['labels_comparison', 'rmsd']].drop_duplicates()
    #     rmsd_data = rmsd_data.sort_values('labels_comparison')
    #     table_data = []
    #     for _, row in rmsd_data.iterrows():
    #         table_data.append([row['labels_comparison'], f"{row['rmsd']:.3f}"])
        
    #     # Add table
    #     if table_data:
    #         table = ax[1].table(cellText=table_data,
    #                           colLabels=['Comparison', 'RMSD (Å)'],
    #                           cellLoc='left',
    #                           loc='center',
    #                           colWidths=[0.7, 0.3])
            
    #         table.auto_set_font_size(False)
    #         table.set_fontsize(9)
    #         table.scale(1, 2)
            
    #         # Style header
    #         for i in range(2):
    #             table[(0, i)].set_facecolor('#E0E0E0')
    #             table[(0, i)].set_text_props(weight='bold')
            
    #         # Color code rows by comparison
    #         for i, (_, row) in enumerate(rmsd_data.iterrows(), start=1):
    #             comparison = row['labels_comparison']
    #             table[(i, 0)].set_facecolor(self.get_palette([comparison])[0])
    #             table[(i, 0)].set_alpha(0.3)
        
    #     ax[1].set_title('RMSD data', fontsize=12, pad=20)
        
    #     # Set figure title
    #     fig.suptitle(name, fontsize=20)
    #     fig.tight_layout()
    #     sns.despine()
        
    #     # Save if path provided
    #     save_path = self.output_dir / save_path if self.output_dir is not None else save_path
    #     plt.savefig(save_path, format='pdf', dpi=300, transparent=True)
        
    #     return fig
    
    def plot_protein_structures(self,
                                     name: str,
                                     save_path: Optional[Path] = None,
                                     figsize: tuple = (20, 15),
                                     image_dpi: int = 150) -> plt.Figure:
        """
        Create a comprehensive PyMOL-based visualization with protein structures and SAXS curves.
        """
        # Define structure types to display
        structure_types = ['out_AF', 'out_NMRtrain', 'out_NMAtrain', 'target']
        structure_labels = [self.map_labels(st) for st in structure_types]
        
        # Get data for this protein
        comparisons = [f"{st}_vs_target" for st in structure_types if st != 'target']
        data = self.df.query('name == @name').copy()
        
        # Prepare data structure
        for col in ['type_a', 'type_b']:
            data[f'labels_{col}'] = data[col].apply(self.map_labels)
        
        # Create figure with 3 rows and 4 columns
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, 
                             height_ratios=[1, 1, 0.8])
        
        # Collect structure files and SAXS data for each type
        structure_data = {}
        for st in structure_types:
            # Find rows where type_a or type_b matches this structure type
            mask_a = data['type_a'] == st
            mask_b = data['type_b'] == st
            
            if mask_a.any():
                row = data[mask_a].iloc[0]
                structure_data[st] = {
                    'fname': row['fname_a'],
                    'saxs_bins': row['saxs_bins_a'],
                    'saxs': row['saxs_a'],
                    'label': self.map_labels(st)
                }
            elif mask_b.any():
                row = data[mask_b].iloc[0]
                structure_data[st] = {
                    'fname': row['fname_b'],
                    'saxs_bins': row['saxs_bins_b'],
                    'saxs': row['saxs_b'],
                    'label': self.map_labels(st)
                }
        
        # Also get alternative structures
        structure_data_alt = {}
        for st in structure_types:
            st_alt = f"{st}_alt"
            mask_a = data['type_a'] == st_alt
            mask_b = data['type_b'] == st_alt
            
            if mask_a.any():
                row = data[mask_a].iloc[0]
                structure_data_alt[st] = {
                    'fname': row['fname_a'],
                    'saxs_bins': row['saxs_bins_a'],
                    'saxs': row['saxs_a'],
                    'label': self.map_labels(st_alt)
                }
            elif mask_b.any():
                row = data[mask_b].iloc[0]
                structure_data_alt[st] = {
                    'fname': row['fname_b'],
                    'saxs_bins': row['saxs_bins_b'],
                    'saxs': row['saxs_b'],
                    'label': self.map_labels(st_alt)
                }
        
        # Row 1: Protein A structures
        for col_idx, st in enumerate(structure_types):
            ax = fig.add_subplot(gs[0, col_idx])
            if st in structure_data:
                img = self._render_pymol_structure(
                    structure_data[st]['fname'],
                    structure_data[st]['label'],
                    dpi=image_dpi
                )
                if img is not None:
                    ax.imshow(img)
                    ax.set_title(f"{structure_data[st]['label']}\n(Protein A)", 
                               fontsize=10, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'Structure\nNot Available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{self.map_labels(st)}\n(Protein A)", fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f"{self.map_labels(st)}\n(Protein A)", fontsize=10)
            ax.axis('off')
        
        # Row 2: Protein B (alternative) structures
        for col_idx, st in enumerate(structure_types):
            ax = fig.add_subplot(gs[1, col_idx])
            if st in structure_data_alt:
                img = self._render_pymol_structure(
                    structure_data_alt[st]['fname'],
                    structure_data_alt[st]['label'],
                    dpi=image_dpi
                )
                if img is not None:
                    ax.imshow(img)
                    ax.set_title(f"{structure_data_alt[st]['label']}\n(Protein B)", 
                               fontsize=10, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'Structure\nNot Available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{self.map_labels(st)} alt\n(Protein B)", fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f"{self.map_labels(st)} alt\n(Protein B)", fontsize=10)
            ax.axis('off')
        
        # Row 3: SAXS curves comparing A and B for each model
        for col_idx, st in enumerate(structure_types):
            ax = fig.add_subplot(gs[2, col_idx])
            
            # Plot SAXS for protein A
            if st in structure_data:
                saxs_bins = structure_data[st]['saxs_bins']
                saxs = structure_data[st]['saxs']
                color_a = self.color_scheme.get(structure_data[st]['label'], '#2b2b2b')
                ax.plot(saxs_bins, saxs, label='Protein A', 
                       color=color_a, linewidth=2)
            
            # Plot SAXS for protein B
            if st in structure_data_alt:
                saxs_bins = structure_data_alt[st]['saxs_bins']
                saxs = structure_data_alt[st]['saxs']
                color_b = self.color_scheme.get(structure_data_alt[st]['label'], '#96acd2')
                ax.plot(saxs_bins, saxs, label='Protein B', 
                       color=color_b, linewidth=2, linestyle='--')
            
            ax.set_xlabel('r (Å)', fontsize=8)
            ax.set_ylabel('P(r)', fontsize=8)
            ax.set_title(f'SAXS: {self.map_labels(st)}', fontsize=9)
            ax.legend(fontsize=7, loc='best')
            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)
        
        # Overall title
        fig.suptitle(f'Protein Structure Comparison: {name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save if path provided
        if save_path is not None:
            save_path = self.output_dir / save_path if self.output_dir is not None else save_path
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        
        return fig
    
    def _render_pymol_structure(self, 
                               pdb_file: Union[str, Path], 
                               label: str,
                               dpi: int = 150,
                               width: int = 1200,
                               height: int = 1200) -> Optional[np.ndarray]:
        """
        Render a protein structure using PyMOL Python API and return as numpy array.
        """
        try:
            import pymol
            from pymol import cmd
        except ImportError:
            print("PyMOL not available. Cannot render structure.")
            return None
            
        # Get color for this structure type
        hex_color = self.color_scheme.get(label, '#2b2b2b')
        # Convert hex to RGB (0-1 range for PyMOL)
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        
        # Create temporary file for the image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_png = tmp_file.name
        
        # Initialize PyMOL in quiet mode
        pymol.finish_launching(['pymol', '-c'])
        
        # Load structure
        cmd.load(str(pdb_file), 'protein')
        
        # Set up cartoon representation
        cmd.hide('everything')
        cmd.show('cartoon', 'protein')
        cmd.set_color('custom_color', [r, g, b])
        cmd.color('custom_color', 'protein')
        
        # Background and rendering settings
        cmd.bg_color('white')
        cmd.set('ray_opaque_background', 0)
        cmd.set('antialias', 2)
        cmd.set('ray_trace_mode', 1)
        
        # Orient and zoom
        cmd.orient('protein')
        cmd.zoom('protein', buffer=2)
        
        # Render with ray tracing
        cmd.ray(width, height)
        
        # Save image
        cmd.png(str(tmp_png), dpi=dpi)
        
        # Clean up PyMOL
        cmd.delete('all')
        cmd.reinitialize()
        
        # Load the rendered image
        if Path(tmp_png).exists():
            img = plt.imread(tmp_png)
            # Clean up temporary file
            Path(tmp_png).unlink()
            return img
        else:
            return None
