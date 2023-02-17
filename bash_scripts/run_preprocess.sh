#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G # memory pool for all cores
#SBATCH --ntasks-per-node=1 # one job per node

#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=t10kw21@abdn.ac.uk 

DIR="/scratch/users/t10kw21/MscProject"
DF_PATH="$DIR/phenotype_files/Phenotypic_V1_0b_preprocessed.csv"
DATA_DIR="$DIR/structural_t1"
SAVE_DIR="$DIR/structural_processed"

module load anaconda3
source activate env_2

srun python3 process_structural.py --df $DF_PATH --data $DATA_DIR --save $SAVE_DIR

