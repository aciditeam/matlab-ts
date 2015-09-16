#!/bin/bash

#SBATCH --partition=mono
#SBATCH --ntasks=1
#SBATCH --time=4-0:00
#SBATCH --mem-per-cpu=8000
#SBATCH -J Deep-DAE_MLP_7_bot_real_RICA_tanh
#SBATCH -e Deep-DAE_MLP_7_bot_real_RICA_tanh.err.txt
#SBATCH -o Deep-DAE_MLP_7_bot_real_RICA_tanh.out.txt

source /etc/profile.modules

module load gcc
module load matlab
cd ~/deepLearn && srun ./deepFunction 7 'DAE' 'MLP' '128   500  1500  1000  2000   250    10' '0  0  0  0  0  0  0' '7_bot_real' 'RICA_tanh' "'iteration.n_epochs', 'learning.lrate', 'use_tanh', 'noise.drop', 'noise.level', 'rica.cost', 'cae.cost'" '200 1e-3 1 0 0 0.1 0' "'iteration.n_epochs', 'use_tanh'" '200 1'