#!/bin/sh
PARENT="$HOME/Documents/MscProject"
DF="$PARENT/phenotype_files/pheno_nn.csv"

DATA_FC="$PARENT/fc_cc200"
DATA_LS="$PARENT/gci_ls_cc200"
DATA_NLS="$PARENT/gci_nls_cc200"

SAVE="$PARENT/vectorised"

python3 vectorise_fc.py --df "$DF" --data "$DATA_LS" --ext "_gci.npy" --save "$SAVE"
python3 vectorise_fc.py --df "$DF" --data "$DATA_NLS" --ext "_gci.npy" --save "$SAVE"

python3 vectorise_fc.py --df "$DF" --data "$DATA_FC" --ext "_fc.npy" --save "$SAVE"



