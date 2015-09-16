#!/bin/bash

#SBATCH --partition=mono
#SBATCH --ntasks=1
#SBATCH --time=4-0:00
#SBATCH --mem-per-cpu=8000
#SBATCH -J Deep-DAE_MLP_6_inc_real_CAE_sig
#SBATCH -e Deep-DAE_MLP_6_inc_real_CAE_sig.err.txt
#SBATCH -o Deep-DAE_MLP_6_inc_real_CAE_sig.out.txt

source /etc/profile.modules

module load gcc
module load matlab
cd ~/deepLearn && srun ./deepFunction 6 'DAE' 'MLP' '128   500  1000  1500  2000    10' '0  0  0  0  0  0' '6_inc_real' 'CAE_sig' "'iteration.n_epochs', 'learning.lrate', 'use_tanh', 'noise.drop', 'noise.level', 'rica.cost', 'cae.cost'" '200 1e-3 0 0 0 0.01 0' "'iteration.n_epochs', 'use_tanh'" '200 0'