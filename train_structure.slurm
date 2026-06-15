#!/bin/bash
#SBATCH --job-name=train_structure
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --qos=shared
#SBATCH --account=m4704

module load python
# Activate your conda environment (edit as needed)
conda activate /pscratch/sd/l/lemonboy/AlphaSAXS_2025

# Set data and output directories (edit as needed)
DATA_DIR=/pscratch/sd/l/lemonboy/scratch_testing/manually_test
CKPT_PATH=/global/cfs/cdirs/m3513/metfish/model_evaluation/all_models_250819/afsaxs_nmr/epoch=15-step=21009.ckpt



# Input CSV as required argument
if [ -z "$1" ]; then
    echo "Error: You must provide an input CSV file as the first argument."
    echo "Usage: sbatch train_structure.slurm <input_csv>"
    exit 1
fi
INPUT_CSV="$1"
# Resolve to absolute path if not already
if [[ "$INPUT_CSV" != /* ]]; then
    INPUT_CSV="$(realpath "$INPUT_CSV")"
fi
CSV_BASENAME=$(basename "$INPUT_CSV" .csv)
OUTPUT_DIR=/pscratch/sd/l/lemonboy/scratch_testing/single_seq_optimization/$CSV_BASENAME
mkdir -p "$OUTPUT_DIR"

python src/metfish/refinement_model/train_structure.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path "$CKPT_PATH" \
    --test_csv_name "$INPUT_CSV" \
    --pdb_ext .pdb \
    --saxs_ext _atom_only.csv \
    --num_iterations 500 \
    --learning_rate 1e-3 \
    --sequence_index 0 \
    --save_frequency 25 \
    --random_init
