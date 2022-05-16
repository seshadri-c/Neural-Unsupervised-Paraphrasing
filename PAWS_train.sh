#!/bin/bash
#SBATCH -A seshadri_c
#SBATCH -c 38
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=paws_op_file.txt
#SBATCH --nodelist=gnode73

source ~/home/Environments/multi_sl/bin/activate

python train_paws.py

