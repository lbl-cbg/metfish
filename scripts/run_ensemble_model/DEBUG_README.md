# Debug Scripts for train_structure.py

After removing redundant files (`new_train.py`, `simple_test.py`), use these scripts to verify everything still works.

## Quick Import Test (Fastest - 10 seconds)

Test all Python imports without running actual training:

```bash
cd /pscratch/sd/l/lemonboy/metfish/scripts/run_ensemble_model
python test_imports.py
```

This checks:
- All required imports work
- `StructureModel` can be instantiated
- Methods like `get_trainable_parameters()` exist

**Expected output:** `SUCCESS: All tests passed!`

---

## Full Debug Run (Medium - 5-10 minutes)

Run a quick 50-iteration optimization on one sequence:

```bash
cd /pscratch/sd/l/lemonboy/metfish/scripts/run_ensemble_model

# Option 1: Use default test protein (1AEL-12_A)
./debug_train_structure.sh

# Option 2: Test specific protein
./debug_train_structure.sh /path/to/your/input.csv
```

**Output location:** `/pscratch/sd/l/lemonboy/metfish/debug_output/train_structure_test_*`

**What it does:**
- Runs 50 iterations (instead of 500) for quick testing
- Saves to separate debug directory (won't overwrite production data)
- Creates timestamped output folder
- Verifies PDB files are generated

**Expected files:**
```
debug_output/train_structure_test_PROTEIN_20251209_HHMMSS/
├── initial/
│   └── PROTEIN.pdb
├── intermediate/
│   ├── PROTEIN_iter_0010.pdb
│   ├── PROTEIN_iter_0020.pdb
│   └── ...
├── best/
│   └── PROTEIN_iter_XXXX.pdb
├── final/
│   └── PROTEIN_optimized.pdb
├── best_model.pth
├── loss_history.npy
└── optimization_summary.txt
```

---

## Production Run (Full - Hours)

Once debug tests pass, run full 500-iteration optimization:

```bash
# Submit to SLURM
cd /pscratch/sd/l/lemonboy/metfish/scripts/run_ensemble_model
sbatch train_structure.slurm /path/to/input.csv
```

---

## Troubleshooting

### Import test fails
```bash
# Check conda environment
conda activate /pscratch/sd/l/lemonboy/AlphaSAXS_2025
which python

# Try importing manually
python -c "from metfish.refinement_model.random_model import StructureModel; print('OK')"
```

### Debug run fails
Check error messages in:
- Terminal output
- SLURM output file (if using sbatch)

Common issues:
- **Missing checkpoint:** Verify `CKPT_PATH` exists
- **Missing data:** Verify `DATA_DIR` and input CSV exist
- **CUDA errors:** Ensure GPU is available (for SLURM runs)

### Files not generated
Check:
```bash
ls -lh /pscratch/sd/l/lemonboy/metfish/debug_output/
```

If empty, check permissions and disk space:
```bash
df -h /pscratch/sd/l/lemonboy/metfish/
```

---

## Key Changes After Cleanup

✅ **Removed:**
- `new_train.py` (unused for paper data generation)
- `simple_test.py` (empty file)

✅ **Kept:**
- `train_structure.py` - main optimization script (YOUR PAPER DATA)
- `save_structure_output()` in `train_structure.py` (as requested)
- `MSARandomModel` in `random_model.py` (used by `generate.py`)
- `compute_plddt_loss()` in `random_model.py` (imported by `train_structure.py`)

✅ **Your production command still works:**
```bash
python $SCRATCH/metfish/src/metfish/refinement_model/train_structure.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path "$CKPT_PATH" \
    --test_csv_name "$INPUT_CSV" \
    --saxs_ext _atom_only.csv \
    --num_iterations 500 \
    --learning_rate 1e-3 \
    --sequence_index 0 \
    --save_frequency 5 \
    --random_init
```
