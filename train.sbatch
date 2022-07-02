#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=CaDeX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=32GB
#SBATCH --time=168:00:00

module load cuda/11.0
cd /usr/stud/srinivaa/code/CaDeX
srun -u /usr/stud/srinivaa/anaconda3/envs/cadex/bin/python3 /usr/stud/srinivaa/code/CaDeX/run.py --config /usr/stud/srinivaa/code/CaDeX/configs/dfaust/training/dfaust_w_pf.yaml -f