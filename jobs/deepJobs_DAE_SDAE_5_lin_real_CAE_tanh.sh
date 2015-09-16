#!/bin/bash

#SBATCH --partition=mono
#SBATCH --ntasks=1
#SBATCH --time=4-0:00
#SBATCH --mem-per-cpu=8000
#SBATCH -J Deep-DAE_SDAE_5_lin_real_CAE_tanh
#SBATCH -e Deep-DAE_SDAE_5_lin_real_CAE_tanh.err.txt
#SBATCH -o Deep-DAE_SDAE_5_lin_real_CAE_tanh.out.txt

source /etc/profile.modules

module load gcc
module load matlab
cd ~/deepLearn && srun ./deepFunction 5 'DAE' 'SDAE' '128  1000  1000  1000    10' '0  0  0  0  0' '5_lin_real' 'CAE_tanh' "'iteration.n_epochs', 'learning.lrate', 'use_tanh', 'noise.drop', 'noise.level', 'rica.cost', 'cae.cost'" '200 1e-3 1 0 0 0.01 0' "'iteration.n_epochs', 'use_tanh'" '200 1'