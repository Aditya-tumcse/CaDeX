#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=CaDeX_dfaust
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2,VRAM:48GB
#SBATCH --mem=128GB
#SBATCH --time=12:00:00

module load cuda/11.0
cd /usr/stud/srinivaa/code/new_CaDeX/CaDeX
srun -u /usr/stud/srinivaa/anaconda3/envs/cadex_1/bin/python3 /usr/stud/srinivaa/code/new_CaDeX/CaDeX/run.py --config /usr/stud/srinivaa/code/new_CaDeX/CaDeX/configs/dfaust/training/dfaust_w_st.yaml -f