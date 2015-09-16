#!/bin/bash

#SBATCH --partition=mono
#SBATCH --ntasks=1
#SBATCH --time=4-0:00
#SBATCH --mem-per-cpu=8000
#SBATCH -J Deep-RBM_MLP_4_dec_real_PERS_base
#SBATCH -e Deep-RBM_MLP_4_dec_real_PERS_base.err.txt
#SBATCH -o Deep-RBM_MLP_4_dec_real_PERS_base.out.txt

source /etc/profile.modules

module load gcc
module load matlab
cd ~/deepLearn && srun ./deepFunction 4 'RBM' 'MLP' '128  1500  1000    10' '0  0  0  0' '4_dec_real' 'PERS_base' "'iteration.n_epochs', 'learning.lrate', 'learning.cd_k', 'learning.persistent_cd', 'parallel_tempering.use'" '200 1e-3 1 1 0' "'iteration.n_epochs'" '200 0'