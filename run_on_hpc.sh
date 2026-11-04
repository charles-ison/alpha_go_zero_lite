#!/bin/bash
#SBATCH -J run_on_HPC
#SBATCH -A eecs 
#SBATCH -p eecs
#SBATCH -t 7-00:00:00   
#SBATCH -c 4
#SBATCH --gres=gpu:1       
#SBATCH --mem=20G  

#SBATCH -o /dev/null
#SBATCH -e models/saved/logs.err

# load env
source env/bin/activate
module load python/3.10 cuda/11.7

python3 -u train.py --game tic_tac_toe
