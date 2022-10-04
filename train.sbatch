#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=CaDeX_exp_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2,VRAM:24GB
#SBATCH --mem=32GB
#SBATCH --time=336:00:00

module load cuda/11.0
cd /usr/stud/srinivaa/code/new_CaDeX/CaDeX
srun -u /usr/stud/srinivaa/anaconda3/envs/cadex_1/bin/python3 /usr/stud/srinivaa/code/new_CaDeX/CaDeX/run.py --config /usr/stud/srinivaa/code/new_CaDeX/CaDeX/configs/dfaust/training/dfaust_w_st.yaml -f