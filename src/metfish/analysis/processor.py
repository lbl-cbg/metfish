import pandas as pd
import numpy as np
import itertools
import mdtraj as md

from scipy.special import rel_entr
from pathlib import Path
from biopandas.pdb import PandasPdb
from typing import List, Dict, Tuple, Optional

from metfish.utils import get_lddt, save_aligned_pdb, get_Pr


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
        # TODO - if we want to do pLDDT analysis, will need to convert bfactors from pdb
        plddt_res_num_a = pdb_df_a.drop_duplicates('residue_number')['residue_number'].to_numpy()
        plddt_res_num_b = pdb_df_b.drop_duplicates('residue_number')['residue_number'].to_numpy()
        plddt_a = pdb_df_a.drop_duplicates('residue_number')['b_factor'].to_numpy()
        plddt_b = pdb_df_b.drop_duplicates('residue_number')['b_factor'].to_numpy()
        
        # Calculate SAXS curves
        r_a, p_of_r_a = get_Pr(fname_a, name, None, 0.5)
        r_b, p_of_r_b = get_Pr(fname_b, name, None, 0.5)
        
        # Calculate SAXS KL divergence
        saxs_kldiv = self._calculate_saxs_kldiv(p_of_r_a, p_of_r_b, r_a, r_b)

        # calculate radius of gyration
        md_obj_a = md.load(fname_a)
        md_obj_b = md.load(fname_b)
        rg_a = md.compute_rg(md_obj_a)[0]
        rg_b = md.compute_rg(md_obj_b)[0]
        rg_diff = rg_a - rg_b

        # Calculate structural metrics
        #rmsd = get_rmsd(pdb_df_a, pdb_df_b, atom_types=['CA'])
        rmsd = md.rmsd(target=md_obj_a,
                       reference=md_obj_b,
                       atom_indices=md_obj_a.topology.select('protein and name CA'),
                       ref_atom_indices=md_obj_b.topology.select('protein and name CA'))[0] * 10.0 # convert to angstrom
        lddt = get_lddt(pdb_df_a, pdb_df_b, atom_types=['CA'])
        
        return {
            'rmsd': rmsd,
            'lddt': lddt,
            'saxs_kldiv': saxs_kldiv,
            'rg_a': rg_a,
            'rg_b': rg_b,
            'rg_diff': rg_diff,
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
                 output_dir: Path,
                 comparison_df_path: Optional[Path] = None):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.aligned_cache_dir = self.output_dir / 'aligned_structures_cache'
        self.aligned_cache_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_df_path = comparison_df_path or (self.output_dir / 'model_comparisons.csv')
        self._comparison_df = None

        self.model_configs = model_dict
        self.tags = {v['tags']: k for k, v in self.model_configs.items()}
        self.analyzer =  ProteinStructureAnalyzer()
        self.apo_holo_pairs = self._get_apo_holo_pairs()

    def _get_apo_holo_pairs(self) -> List[Tuple[str, str]]:
        """Extract apo-holo structure pairs from metadata."""
        apo_holo_csv_name = self.data_dir / "Table_rmsd_Apo_vs_Holo.csv"
        if apo_holo_csv_name.exists():
            metadata_df = pd.read_csv(apo_holo_csv_name, delimiter=';')
            return list(zip(metadata_df['Apo_ID'], metadata_df['Holo_ID']))
            
        return None
    
    def _create_comparisons_list(self) -> List[Tuple[str, str]]:
        comparisons=[('out', 'target'), ('out', 'out_alt'), ('target', 'target_alt')]
        all_comparisons = [
            tuple(c.replace('out', f'out_{t}') for c in comp)
            for comp in comparisons
            if any('out' == x or 'out_alt' == x for x in comp)
            for t in self.tags
        ]
        all_comparisons.extend([tuple(f'out_{x}' for x in comb) for comb in itertools.combinations(self.tags, 2)])
        return all_comparisons

    def get_comparison_df(self,
                          names: List[str] = None,
                          overwrite: bool = False) -> pd.DataFrame:
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
        if self._comparison_df is not None and not overwrite:
            return self._comparison_df

        # Try to load from cache
        if not overwrite and self.comparison_df_path.exists():
            print(f"Loading cached comparison dataframe from {self.comparison_df_path}")
            self._comparison_df = pd.read_csv(self.comparison_df_path)
            if self._comparison_df is not None:
                return self._comparison_df
        
        # default to recreating comparison
        self._comparison_df = self._create_comparison_df(names)
        self._comparison_df.to_csv(self.comparison_df_path, index=False)

        return self._comparison_df

    def _create_comparison_df(self, names: List[str]) -> pd.DataFrame:
        """Create comparison dataframe from scratch."""
        comparisons = self._create_comparisons_list()
        data = []

        for name in names:
            print(f'Generating comparison data for {name}...')
            
            # Get name of the other pair item
            name_alt = self._get_pair_name(name, names)
            if name_alt is None or name_alt not in names:
                continue
                
            # Process all comparisons
            for (a, b) in comparisons:
                comparison_data = self._process_single_comparison(
                    name, name_alt, a, b,
                )
                if comparison_data:
                    data.append(comparison_data)
        
        return pd.DataFrame(data)
    
    def _get_pair_name(self, 
                       name: str, 
                       names: List[str]) -> Optional[str]:
        """Find alternative structure name from pairs."""
        if self.apo_holo_pairs is not None:
            for p in self.apo_holo_pairs:
                if name in p:
                    name_alt = (set(p) - {name}).pop()
                    return name_alt
        return name
    
    def _process_single_comparison(self,
                                   name: str,
                                   name_alt: str,
                                   type_a: str,
                                   type_b: str) -> Optional[Dict]:
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
            save_aligned_pdb(fname_a, fname_b, comparison, self.aligned_cache_dir)
            
            # Compile results
            return {
                'name': name,
                'name_alt': name_alt,
                'type_a': type_a,
                'type_b': type_b,
                'comparison': comparison,
                'fname_a': fname_a,
                'fname_b': fname_b,
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
        tags_a = next((t for t in self.tags.keys() if t in type_a), '')
        tags_b = next((t for t in self.tags.keys() if t in type_b), '')

        dir_path_a = f'{self.data_dir}/pdbs' if 'target' in type_a else f'{self.output_dir}/{tags_a}/'
        dir_path_b = f'{self.data_dir}/pdbs' if 'target' in type_b else f'{self.output_dir}/{tags_b}/'
        
        name_a = name_alt if 'alt' in type_a else name
        name_b = name_alt if 'alt' in type_b else name
        
        fname_ext_a = self.get_filename_ext(type_a)
        fname_ext_b = self.get_filename_ext(type_b)
        
        fname_a = f"{dir_path_a}/{name_a}{fname_ext_a}"
        fname_b = f"{dir_path_b}/{name_b}{fname_ext_b}"
        
        return fname_a, fname_b
    
    def get_filename_ext(self, tag: str) -> str:
        """Get filename extension based on model tag."""        
        for k, v in self.tags.items():
            if f"out_{k}" == tag or k in tag:
                return f"_{self.model_configs[v]['model_name']}_{k}.pdb"
            elif 'target' in tag:
                return '.pdb'
