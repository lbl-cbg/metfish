#!/usr/bin/env python3
"""Quick test script - Test Figure 2 (no ensembles needed)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metfish.analysis.analysis_draft import ApoHoloPairAnalyzer, DataPaths

# Test Figure 2 with just 3 proteins (no ensembles)
analyzer = ApoHoloPairAnalyzer()
all_pdbs = [f.stem for f in Path(DataPaths.REF).iterdir() if f.suffix == '.pdb']
pdb_list = all_pdbs[:3]  # Take first 3

print(f"Testing Figure 2 with: {pdb_list}")
df = analyzer.calculate_metrics(pdb_list)
print(f"\nResult:\n{df}")
print(f"\nShape: {df.shape}")
