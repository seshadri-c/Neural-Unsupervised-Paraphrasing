#!/bin/bash
#SBATCH -A research
#SBATCH -c 38
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=msr_op_file.txt
#SBATCH --nodelist=gnode59

source ~/home/Environments/multi_sl/bin/activate

python train_msr.py

