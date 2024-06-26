#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=1:30:00
#SBATCH --constraint=gpu
#SBATCH --ntasks 64 
#SBATCH --ntasks-per-node 4 
#SBATCH --gpus-per-node 4
#SBATCH -o ./%x_logs/%j.log
#SBATCH -e ./%x_logs/%j.log
#SBATCH -J msa_model

module load python
module load cudatoolkit/11.5
module load gcc/11.2.0

conda activate alphaflow

DATA_DIR="/global/cfs/cdirs/m3513/metfish/PDB70_verB_fixed_data/result"
OUTPUT_DIR="/pscratch/sd/s/smprince/projects/metfish/model_outputs"
TRAIN_SCRIPT="/pscratch/sd/s/smprince/projects/metfish/src/metfish/msa_model/train.py"
GPUS_PER_NODE=4 
NUM_NODES=16

# -n must match num_nodes * gpus-per-node indicated in training script
srun -n 64 python $TRAIN_SCRIPT $DATA_DIR $OUTPUT_DIR --gpus_per_node $GPUS_PER_NODE --num_nodes $NUM_NODES
# srun -n 64 python $TRAIN_SCRIPT $DATA_DIR $OUTPUT_DIR --gpus_per_node $GPUS_PER_NODE --num_nodes $NUM_NODES --validate_only