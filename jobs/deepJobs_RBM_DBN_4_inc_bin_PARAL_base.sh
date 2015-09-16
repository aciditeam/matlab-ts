#!/bin/bash

#SBATCH --partition=mono
#SBATCH --ntasks=1
#SBATCH --time=4-0:00
#SBATCH --mem-per-cpu=8000
#SBATCH -J Deep-RBM_DBN_4_inc_bin_PARAL_base
#SBATCH -e Deep-RBM_DBN_4_inc_bin_PARAL_base.err.txt
#SBATCH -o Deep-RBM_DBN_4_inc_bin_PARAL_base.out.txt

source /etc/profile.modules

module load gcc
module load matlab
cd ~/deepLearn && srun ./deepFunction 4 'RBM' 'DBN' '128  1000  1500    10' '0  1  1  1' '4_inc_bin' 'PARAL_base' "'iteration.n_epochs', 'learning.lrate', 'learning.cd_k', 'learning.persistent_cd', 'parallel_tempering.use'" '200 1e-3 1 0 1' "'iteration.n_epochs', 'learning.persistent_cd'" '200 0'