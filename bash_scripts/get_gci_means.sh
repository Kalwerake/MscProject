#!/bin/sh
PARENT="$HOME/Documents/MscProject"
DF="$PARENT/phenotype_files/pheno_nn.csv"

DATA_LS="$PARENT/gci_ls_cc200"
DATA_NLS="$PARENT/gci_nls_cc200"

python3 gci_mean.py --df "$DF" --data "$DATA_LS" --ext "_gci.npy" --binary
python3 gci_mean.py --df "$DF" --data "$DATA_LS" --ext "_gci.npy" --no-binary

python3 gci_mean.py --df "$DF" --data "$DATA_NLS" --ext "_gci.npy" --binary
python3 gci_mean.py --df "$DF" --data "$DATA_NLS" --ext "_gci.npy" --no-binary


