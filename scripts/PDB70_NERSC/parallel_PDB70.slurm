#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --account=m3513

module load python
module load parallel
conda activate /pscratch/sd/l/lemonboy/PDB_prep
ls  *.pdb | parallel -j 4 bash fixpdb_pr_parallel.bash -f {} -o . :::