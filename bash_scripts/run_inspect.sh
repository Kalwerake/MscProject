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
DATA_DIR="$DIR/structural_processed"
FAIL="$DATA_DIR/fail.txt"
DF_PATH="$DIR/phenotype_files/Phenotypic_V1_0b_preprocessed.csv"
SAVE="$DIR/phenotype_files/structural_nn.csv"

module load anaconda3
source activate env_2

srun python3 inspect_processed.py --fail $FAIL --df $DF_PATH --data $DATA_DIR --save $SAVE
