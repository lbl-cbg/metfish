#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --account=m3513

module load python
module load parallel
conda activate /pscratch/sd/l/lemonboy/PDB_fixer_test
find ./ -maxdepth 1 -name '*.pdb' -type f | parallel -j 32 bash fixpdb_pr_parallel.bash -f {} -o ../result
