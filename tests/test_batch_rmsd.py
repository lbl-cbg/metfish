#!/usr/bin/env python
"""
Test to verify that batch RMSD calculation gives identical results to individual RMSD.

This test compares:
1. Individual RMSD: Loading conformations one-by-one and computing RMSD separately
2. Batch RMSD: Loading all conformations at once and computing RMSD vectorized

Expected: Both methods should give identical RMSD values (within floating point precision)
"""
import mdtraj as md
import numpy as np
import os
import sys
import time

sys.path.insert(0, 'src')
from metfish.analysis.analysis_draft import DataPaths

def test_batch_vs_individual_rmsd(pdb_id='1EX6_B', n_conformations=10):
    """
    Compare batch and individual RMSD calculations.
    
    Parameters
    ----------
    pdb_id : str
        Protein ID to test
    n_conformations : int
        Number of conformations to test (use smaller number for speed)
    """
    print("=" * 80)
    print("TESTING: Batch RMSD vs Individual RMSD")
    print("=" * 80)
    print(f"\nProtein: {pdb_id}")
    print(f"Number of conformations: {n_conformations}\n")
    
    # Load reference
    ref_path = os.path.join(DataPaths.REF, f'{pdb_id}.pdb')
    ref_traj = md.load(ref_path)
    ref_ca = ref_traj.topology.select('name CA')
    print(f"Reference loaded: {ref_traj.n_residues} residues, {len(ref_ca)} CA atoms")
    
    # Get conformation paths
    import pandas as pd
    csv_path = os.path.join(DataPaths.ENSEMBLE, pdb_id, 'saved_structures.csv')
    df = pd.read_csv(csv_path)
    df = df[~df['full_path'].str.contains('best')]
    df = df.sort_values('loss').head(n_conformations)
    pdb_paths = df['full_path'].tolist()
    print(f"Found {len(pdb_paths)} conformation files\n")
    
    # =========================================================================
    # METHOD 1: Individual RMSD (one-by-one loading)
    # =========================================================================
    print("-" * 80)
    print("METHOD 1: Individual RMSD (loading one-by-one)")
    print("-" * 80)
    
    individual_rmsds = []
    t0 = time.time()
    
    for i, pdb_path in enumerate(pdb_paths):
        traj = md.load(pdb_path)
        traj_ca = traj.topology.select('name CA')
        
        if len(traj_ca) != len(ref_ca):
            print(f"  WARNING: Conf {i} has {len(traj_ca)} CA atoms, expected {len(ref_ca)}")
            continue
        
        rmsd = md.rmsd(traj, ref_traj, 0, traj_ca, ref_ca)
        individual_rmsds.append(rmsd[0] * 10)  # Convert nm to Angstrom
    
    t1 = time.time()
    individual_time = t1 - t0
    
    print(f"Completed: {len(individual_rmsds)} RMSDs calculated")
    print(f"Time: {individual_time:.3f}s ({individual_time/n_conformations:.3f}s per conformation)")
    print(f"Sample RMSDs: {individual_rmsds[:5]}\n")
    
    # =========================================================================
    # METHOD 2: Batch RMSD (load all at once, vectorized calculation)
    # =========================================================================
    print("-" * 80)
    print("METHOD 2: Batch RMSD (vectorized calculation)")
    print("-" * 80)
    
    t0 = time.time()
    
    # Load all conformations at once
    ensemble_traj = md.load(pdb_paths)
    tl = time.time()
    
    # Vectorized RMSD calculation for ALL frames at once
    ensemble_ca = ensemble_traj.topology.select('name CA')
    
    if len(ensemble_ca) != len(ref_ca):
        print(f"  ERROR: Ensemble has {len(ensemble_ca)} CA atoms, expected {len(ref_ca)}")
        return False
    
    rmsd_values = md.rmsd(ensemble_traj, ref_traj, 0, ensemble_ca, ref_ca)
    batch_rmsds = rmsd_values * 10  # Convert nm to Angstrom
    
    t1 = time.time()
    batch_time = t1 - t0
    load_time = tl - t0
    calc_time = t1 - tl
    
    print(f"Completed: {len(batch_rmsds)} RMSDs calculated")
    print(f"Time breakdown:")
    print(f"  - Batch loading: {load_time:.3f}s")
    print(f"  - Batch RMSD calc: {calc_time:.3f}s")
    print(f"  - Total: {batch_time:.3f}s ({batch_time/n_conformations:.3f}s per conformation)")
    print(f"Sample RMSDs: {batch_rmsds[:5].tolist()}\n")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    individual_rmsds = np.array(individual_rmsds)
    
    # Check if arrays have same length
    if len(individual_rmsds) != len(batch_rmsds):
        print(f"❌ MISMATCH: Different number of results!")
        print(f"   Individual: {len(individual_rmsds)}")
        print(f"   Batch: {len(batch_rmsds)}")
        return False
    
    # Check if values are identical (within floating point precision)
    differences = np.abs(individual_rmsds - batch_rmsds)
    max_diff = np.max(differences)
    mean_diff = np.mean(differences)
    
    print(f"\nRMSD Value Comparison:")
    print(f"  Maximum difference: {max_diff:.10f} Å")
    print(f"  Mean difference:    {mean_diff:.10f} Å")
    print(f"  Relative error:     {max_diff / np.mean(individual_rmsds) * 100:.6f}%")
    
    # Tolerance check (should be within numerical precision)
    tolerance = 1e-6  # 1 microangstrom
    
    if max_diff < tolerance:
        print(f"\n✅ PASS: Values are identical within tolerance ({tolerance} Å)")
        results_match = True
    else:
        print(f"\n❌ FAIL: Differences exceed tolerance ({tolerance} Å)")
        print("\nFirst 10 comparisons:")
        for i in range(min(10, len(individual_rmsds))):
            print(f"  Conf {i}: Individual={individual_rmsds[i]:.6f}Å  "
                  f"Batch={batch_rmsds[i]:.6f}Å  Diff={differences[i]:.9f}Å")
        results_match = False
    
    # Speed comparison
    print(f"\nSpeed Comparison:")
    print(f"  Individual method: {individual_time:.3f}s")
    print(f"  Batch method:      {batch_time:.3f}s")
    speedup = individual_time / batch_time
    print(f"  Speedup:           {speedup:.2f}x faster")
    
    if speedup > 1.0:
        print(f"\n✅ Batch method is {speedup:.2f}x faster!")
    else:
        print(f"\n⚠️  Batch method is slower (ratio: {speedup:.2f})")
    
    print("\n" + "=" * 80)
    
    return results_match


if __name__ == "__main__":
    # Test with different numbers of conformations
    test_cases = [
        ('1EX6_B', 10),   # Small test
        ('1EX6_B', 50),   # Full ensemble
    ]
    
    all_passed = True
    
    for pdb_id, n_conf in test_cases:
        print(f"\n\n")
        passed = test_batch_vs_individual_rmsd(pdb_id, n_conf)
        all_passed = all_passed and passed
        print("\n")
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! Batch RMSD gives identical results.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED! Check output above.")
        sys.exit(1)
