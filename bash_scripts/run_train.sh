#!/bin/bash
#SBATCH --nodes=1 # number of nodes
#SBATCH --cpus-per-task=20 # number of cores
#SBATCH --mem=100G # memory pool for all cores
#SBATCH --time=250:00:00
#SBATCH --ntasks-per-node=1 # one job per node
#SBATCH --partition=gpu

#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err 

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=t10kw21@abdn.ac.uk 

DIR="/home/t10kw21/sharedscratch/MscProject"
DF_PATH="$DIR/phenotype_files"
TRAIN="$DF_PATH/train_surr.csv"
TEST="$DF_PATH/test_surr.csv"
DATA_DIR="$DIR/dfc_aug"
SAVE_MODEL="$DIR/model_params/CRNN5.pth"
SAVE_METRICS="$DIR/model_metrics/CRNN5_metrics.pkl"

module load anaconda3
source activate env_2

srun python3 train_model.py --train $TRAIN --test $TEST \
--data $DATA_DIR --batch 256 --lr 0.0001 --epochs 100 --workers 10 \
--model_save $SAVE_MODEL --metric_save $SAVE_METRICS --early_stop \
--patience 10 --delta 4.5   
