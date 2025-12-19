#!/bin/bash
# Debug script to test train_structure.py on a single sequence
# Usage: bash debug_train_structure.sh [input_csv]

set -e  # Exit on error

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate /pscratch/sd/l/lemonboy/AlphaSAXS_2025

# Set directories
DATA_DIR=/global/cfs/cdirs/m4704/100125_Nature_Com_data/Apo_holo_data
CKPT_PATH=/global/cfs/cdirs/m4704/100125_Nature_Com_data/ensemble_generated/checkpoint/epoch=15-step=21009.ckpt

# Input CSV - use provided argument or default test file
if [ -z "$1" ]; then
    echo "No input CSV provided, using default test file..."
    INPUT_CSV="$DATA_DIR/input_csv_split/1AEL-12_A.csv"  # Use first protein as test
else
    INPUT_CSV="$1"
fi

# Resolve to absolute path
if [[ "$INPUT_CSV" != /* ]]; then
    INPUT_CSV="$(realpath "$INPUT_CSV")"
fi

CSV_BASENAME=$(basename "$INPUT_CSV" .csv)

# DEBUG OUTPUT DIR - separate location to avoid overwriting production data
OUTPUT_DIR=/pscratch/sd/l/lemonboy/metfish/debug_output/train_structure_test_${CSV_BASENAME}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "DEBUG TEST: train_structure.py"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Checkpoint: $CKPT_PATH"
echo "Input CSV: $INPUT_CSV"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Verify files exist
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: Input CSV not found: $INPUT_CSV"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Run with reduced iterations for quick test
echo "Starting optimization test (50 iterations for quick debug)..."
python $SCRATCH/metfish/src/metfish/refinement_model/train_structure.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path "$CKPT_PATH" \
    --test_csv_name "$INPUT_CSV" \
    --saxs_ext _atom_only.csv \
    --num_iterations 50 \
    --learning_rate 1e-3 \
    --sequence_index 0 \
    --save_frequency 10 \
    --random_init

# Check if output files were created
echo ""
echo "=========================================="
echo "DEBUG TEST RESULTS"
echo "=========================================="
if [ -d "$OUTPUT_DIR" ]; then
    echo "✓ Output directory created: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    find "$OUTPUT_DIR" -type f -name "*.pdb" -o -name "*.txt" -o -name "*.npy" -o -name "*.pth" | head -20
    echo ""
    echo "Directory structure:"
    tree -L 2 "$OUTPUT_DIR" 2>/dev/null || ls -R "$OUTPUT_DIR"
    echo ""
    echo "✓ Test completed successfully!"
    echo "Review results in: $OUTPUT_DIR"
else
    echo "✗ ERROR: Output directory not created"
    exit 1
fi

echo ""
echo "To run full optimization (500 iterations), use:"
echo "python \$SCRATCH/metfish/src/metfish/refinement_model/train_structure.py \\"
echo "    --data_dir \"$DATA_DIR\" \\"
echo "    --output_dir \"/path/to/full/output\" \\"
echo "    --ckpt_path \"$CKPT_PATH\" \\"
echo "    --test_csv_name \"$INPUT_CSV\" \\"
echo "    --saxs_ext _atom_only.csv \\"
echo "    --num_iterations 500 \\"
echo "    --learning_rate 1e-3 \\"
echo "    --sequence_index 0 \\"
echo "    --save_frequency 5 \\"
echo "    --random_init"
