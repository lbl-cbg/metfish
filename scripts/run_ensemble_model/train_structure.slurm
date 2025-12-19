#!/bin/bash
#SBATCH --job-name=train_structure
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q shared
#SBATCH -t 2:0:0
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --account=m4704_g

module load python
# Activate your conda environment (edit as needed)
conda activate /pscratch/sd/l/lemonboy/AlphaSAXS_2025

# Set data and output directories (edit as needed)
DATA_DIR=/global/cfs/cdirs/m4704/100125_Nature_Com_data/Apo_holo_data
CKPT_PATH=/global/cfs/cdirs/m4704/100125_Nature_Com_data/ensemble_generated/checkpoint/epoch=15-step=21009.ckpt

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
OUTPUT_DIR=/global/cfs/cdirs/m4704/100125_Nature_Com_data/ensemble_generated/ensembles/$CSV_BASENAME
mkdir -p "$OUTPUT_DIR"

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
