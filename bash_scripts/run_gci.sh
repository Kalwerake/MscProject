#!/bin/sh
PARENT="$HOME/Documents/MscProject"
DATA="$PARENT/rois_cc200"
SAVE_LS="$PARENT/gci_ls_cc200"
SAVE_NLS="$PARENT/gci_nls_cc200"
DF="$PARENT/phenotype_files/pheno_nn.csv"

python3 granger_calculate.py --df "$DF" --data "$DATA" --save "$SAVE_LS" --suffix "_rois_cc200.1D" --large
python3 granger_calculate.py --df "$DF" --data "$DATA" --save "$SAVE_NLS" --suffix "_rois_cc200.1D" --no-large
