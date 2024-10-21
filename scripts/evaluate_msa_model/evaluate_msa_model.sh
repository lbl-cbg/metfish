#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=01:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks 1 
#SBATCH --ntasks-per-node 1 
#SBATCH --account=m3513_g
#SBATCH -o ./%x_logs/%j.log
#SBATCH -e ./%x_logs/%j.log
#SBATCH -J msa_saxs_evaluation
#SBATCH --mail-user=smprince@lbl.gov
#SBATCH --mail-type=ALL

module load python
module load cudatoolkit/11.5
module load gcc/11.2.0

conda activate alphaflow

PREDICT_SCRIPT="/pscratch/sd/s/smprince/projects/metfish/scripts/evaluate_msa_model/evaluate_msa_model.py"
srun python $PREDICT_SCRIPT