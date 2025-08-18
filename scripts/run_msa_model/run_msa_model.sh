#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=00:05:00
#SBATCH --constraint=gpu
#SBATCH --ntasks 64 
#SBATCH --ntasks-per-node 4 
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --account=m4704_g
#SBATCH -o ./%x_logs/%j.log
#SBATCH -e ./%x_logs/%j.log
#SBATCH --job-name afsaxs_nmr
#SBATCH --mail-user=smprince@lbl.gov
#SBATCH --mail-type=ALL

# load necessary modules and environment
module load python
module load cudatoolkit/11.7
module load gcc/11.2.0

conda activate /pscratch/sd/s/smprince/projects/metfish/alphaflow

# default values
DATASET="nmr"
OUTPUT_DIR="/pscratch/sd/s/smprince/projects/metfish/model_outputs/"
TRAIN_SCRIPT="/pscratch/sd/s/smprince/projects/metfish/src/metfish/msa_model/train.py"
GPUS_PER_NODE=4 
NUM_NODES=16
TOTAL_TASKS=$((GPUS_PER_NODE * NUM_NODES))

# specify checkpoint if desired
CKPT_PATH="/pscratch/sd/s/smprince/projects/metfish/model_outputs/checkpoints/afsaxs_nmr/epoch=4-step=6565.ckpt"
CKPT_ARGS=""
if [[ -n "$CKPT_PATH" ]]; then
    CKPT_ARGS="--resume_from_ckpt --ckpt_path $CKPT_PATH"
fi

# determine dataset to use
case $DATASET in
    nmr)
        DATA_DIR="/global/cfs/cdirs/m3513/metfish/NMR_training/data_for_training"
        ;;
    nma)
        DATA_DIR="/global/cfs/cdirs/m3513/metfish/PDB70_ANM_simulated_data"
        ;;
    pdb70)
        DATA_DIR="/global/cfs/cdirs/m3513/metfish/PDB70_verB_fixed_data/result"
        ;;
    *)
        echo "Unknown dataset: $DATASET. Use nmr, nma, or pdb70"
        exit 1
        ;;
esac


# running with AF weights unfrozen
srun -n $TOTAL_TASKS python $TRAIN_SCRIPT $DATA_DIR $OUTPUT_DIR --gpus_per_node $GPUS_PER_NODE --num_nodes $NUM_NODES --job_name $SLURM_JOB_NAME $CKPT_ARGS --use_l1_loss --unfreeze_af_weights --max_epochs 20

# run validation only
# srun -n $TOTAL_TASKS python $TRAIN_SCRIPT $DATA_DIR $OUTPUT_DIR --gpus_per_node $GPUS_PER_NODE --num_nodes $NUM_NODES --job_name $SLURM_JOB_NAME $CKPT_ARGS --use_l1_loss --validate_only

# running from scratch
#srun -n 64 python $TRAIN_SCRIPT $DATA_DIR $OUTPUT_DIR --gpus_per_node $GPUS_PER_NODE --num_nodes $NUM_NODES --job_name $SLURM_JOB_NAME $CKPT_ARGS --use_l1_loss --unfreeze_af_weights --jax_param_path ""