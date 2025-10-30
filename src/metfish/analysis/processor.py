import pandas as pd
import numpy as np
from scipy.special import rel_entr
from pathlib import Path
from biopandas.pdb import PandasPdb
from typing import List, Dict, Tuple, Optional

from metfish.utils import get_rmsd, get_lddt, save_aligned_pdb, get_Pr


class ProteinStructureAnalyzer:
    """Analyze and compare protein structures."""

    def calculate_metrics(self, 
                         pdb_df_a: pd.DataFrame, 
                         pdb_df_b: pd.DataFrame,
                         fname_a: str,
                         fname_b: str,
                         name: str) -> Dict:
        """Calculate comparison metrics between two structures."""
        
        # Extract pLDDT values
        plddt_res_num_a = pdb_df_a.drop_duplicates('residue_number')['residue_number'].to_numpy()
        plddt_res_num_b = pdb_df_b.drop_duplicates('residue_number')['residue_number'].to_numpy()
        plddt_a = pdb_df_a.drop_duplicates('residue_number')['b_factor'].to_numpy()
        plddt_b = pdb_df_b.drop_duplicates('residue_number')['b_factor'].to_numpy()
        
        # Calculate SAXS curves
        r_a, p_of_r_a = get_Pr(fname_a, name, None, 0.5)
        r_b, p_of_r_b = get_Pr(fname_b, name, None, 0.5)
        
        # Calculate SAXS KL divergence
        saxs_kldiv = self._calculate_saxs_kldiv(p_of_r_a, p_of_r_b, r_a, r_b)
        
        # Calculate structural metrics
        rmsd = get_rmsd(pdb_df_a, pdb_df_b, atom_types=['CA'])
        lddt = get_lddt(pdb_df_a, pdb_df_b, atom_types=['CA'])
        
        return {
            'rmsd': rmsd,
            'lddt': lddt,
            'saxs_kldiv': saxs_kldiv,
            'plddt_a': plddt_a,
            'plddt_bins_a': plddt_res_num_a,
            'plddt_b': plddt_b,
            'plddt_bins_b': plddt_res_num_b,
            'saxs_bins_a': r_a,
            'saxs_a': p_of_r_a,
            'saxs_bins_b': r_b,
            'saxs_b': p_of_r_b,
        }
    
    def _calculate_saxs_kldiv(self, 
                             p_of_r_a: np.ndarray, 
                             p_of_r_b: np.ndarray,
                             r_a: np.ndarray,
                             r_b: np.ndarray) -> float:
        """Calculate KL divergence between SAXS profiles."""
        eps = 1e-10
        max_len = max(len(r_a), len(r_b))
        
        saxs_a_padded = np.pad(p_of_r_a, (0, max_len - len(r_a)), 
                              mode='constant', constant_values=0)
        saxs_b_padded = np.pad(p_of_r_b, (0, max_len - len(r_b)), 
                              mode='constant', constant_values=0)
        
        saxs_a_padded = (saxs_a_padded + eps) / np.sum(saxs_a_padded + eps)
        saxs_b_padded = (saxs_b_padded + eps) / np.sum(saxs_b_padded + eps)
        
        return np.sum(rel_entr(saxs_a_padded, saxs_b_padded))

class ModelComparisonProcessor:
    """Process and create comparison dataframes for multiple models."""
    
    def __init__(self, 
                 model_dict: Dict[str, Dict],
                 data_dir: Path,
                 output_dir: Path):
        self.model_configs = model_dict
        self.analyzer =  ProteinStructureAnalyzer()
        self.data_dir = data_dir
        self.output_dir = output_dir
        
    def create_comparison_df(self,
                            pairs: List[Tuple[str, str]],
                            names: List[str],
                            comparisons: List[Tuple[str, str]],
                            metadata_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create comprehensive comparison dataframe.
        
        Args:
            pairs: List of structure pairs to compare
            names: List of structure names to process
            comparisons: List of comparison types (e.g., ('out', 'target'))
            metadata_df: Optional metadata dataframe with RMSD info
            
        Returns:
            DataFrame with all comparison metrics
        """
        data = []
        
        for name in names:
            print(f'Generating comparison data for {name}...')
            
            # Get alternative name from pairs
            name_alt = self._get_alternative_name(name, pairs, names)
            if name_alt is None:
                continue
                
            # Load metadata if available
            rmsd_metadata = self._get_metadata(name, metadata_df) if metadata_df is not None else None
            
            # Process all comparisons
            for (a, b) in comparisons:
                comparison_data = self._process_single_comparison(
                    name, name_alt, a, b, rmsd_metadata
                )
                if comparison_data:
                    data.append(comparison_data)
        
        return pd.DataFrame(data)
    
    def _get_alternative_name(self, 
                             name: str, 
                             pairs: List[Tuple[str, str]], 
                             names: List[str]) -> Optional[str]:
        """Find alternative structure name from pairs."""
        for p in pairs:
            if p[0] == p[1]:
                return name
            elif name in p:
                name_alt = (set(p) - {name}).pop()
                if name_alt in names:
                    return name_alt
        return None
    
    def _get_metadata(self, name: str, metadata_df: pd.DataFrame) -> Optional[float]:
        """Extract metadata for a given structure."""
        result = metadata_df.query('apo_id == @name | holo_id == @name')['rmsd_apo_holo']
        return result.values[0] if not result.empty else None
    
    def _process_single_comparison(self,
                                   name: str,
                                   name_alt: str,
                                   type_a: str,
                                   type_b: str,
                                   rmsd_metadata: Optional[float]) -> Optional[Dict]:
        """Process a single structure comparison."""
        try:
            # Get file paths
            fname_a, fname_b = self._get_comparison_files(
                name, name_alt, type_a, type_b
            )
            
            # Load PDB data
            pdb_df_a = PandasPdb().read_pdb(fname_a).df['ATOM']
            pdb_df_b = PandasPdb().read_pdb(fname_b).df['ATOM']
            
            # Calculate metrics
            metrics = self.analyzer.calculate_metrics(
                pdb_df_a, pdb_df_b, fname_a, fname_b, name
            )
            
            # Save aligned structure
            comparison = f'{type_a}_vs_{type_b}'
            save_aligned_pdb(fname_a, fname_b, comparison)
            # TOOO - check where these output files are going
            
            # Compile results
            return {
                'name': name,
                'name_alt': name_alt,
                'type_a': type_a,
                'type_b': type_b,
                'comparison': comparison,
                'fname_a': fname_a,
                'fname_b': fname_b,
                'rmsd_apo_holo': rmsd_metadata,
                **metrics
            }
            
        except FileNotFoundError as e:
            print(f"File not found: {e}. Skipping...")
            return None
    
    def _get_comparison_files(self,
                             name: str,
                             name_alt: str,
                             type_a: str,
                             type_b: str) -> Tuple[str, str]:
        """Get file paths for comparison structures."""
        dir_path_a = f'{self.data_dir}/pdbs' if 'target' in type_a else self.output_dir
        dir_path_b = f'{self.data_dir}/pdbs' if 'target' in type_b else self.output_dir
        
        name_a = name_alt if 'alt' in type_a else name
        name_b = name_alt if 'alt' in type_b else name
        
        fname_ext_a = self.get_filename_ext(type_a)
        fname_ext_b = self.get_filename_ext(type_b)
        
        fname_a = f"{dir_path_a}/{name_a}{fname_ext_a}"
        fname_b = f"{dir_path_b}/{name_b}{fname_ext_b}"
        
        return fname_a, fname_b
    
    def get_filename_ext(self, tag: str) -> str:
        """Get filename extension based on model tag."""
        tags_to_keys = {v['tags']: k for k, v in self.model_configs.items()}
        
        for k, v in tags_to_keys.items():
            if f"out_{k}" == tag:
                return f"_{self.model_configs[v]['model_name']}_{k}_unrelaxed.pdb"
            elif k in tag:
                return f"_{self.model_configs[v]['model_name']}_{k}_unrelaxed.pdb"
            elif 'target' in tag:
                return '.pdb'
        
        raise ValueError(f"Unknown tag: {tag}")