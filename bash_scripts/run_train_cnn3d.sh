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
TRAIN="$DF_PATH/train_mri.csv"
TEST="$DF_PATH/test_mri.csv"
DATA_DIR="$DIR/structural_processed"
SAVE_MODEL="$DIR/model_params/CNN3D.pth"
SAVE_METRICS="$DIR/model_metrics/CNN3D_metrics.pkl"

module load anaconda3
source activate env_2

srun python3 train_model_cnn.py --train $TRAIN --test $TEST \
--data $DATA_DIR --batch 1 --lr 0.0001 --epochs 50 --workers 10 \
--model_save $SAVE_MODEL --metric_save $SAVE_METRICS --no-early_stop
