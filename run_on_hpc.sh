#!/bin/bash
#SBATCH -J run_on_HPC
#SBATCH -A eecs 
#SBATCH -p dgx
#SBATCH -t 7-00:00:00   
#SBATCH -c 4
#SBATCH --gres=gpu:1       
#SBATCH --mem=20G  

#SBATCH -o neural_networks/saved_models/logs.out
#SBATCH -e neural_networks/saved_models/logs.err

# load env
source env/bin/activate
module load python/3.10 cuda/11.7

python3 -u -m tic_tac_toe.train_for_tic_tac_toe.py
