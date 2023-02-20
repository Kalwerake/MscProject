#!/bin/sh


PARENT="$HOME/Documents/MscProject"
DF="$PARENT/phenotype_files/pheno_nn.csv"
DATA="$PARENT/rois_cc200"
EXT="_rois_cc200.1D"

python3 fc_maker.py --df "$DF" --data "$DATA" --extension $EXT --save "$PARENT/fc_cc200"
python3 dfc_maker.py --df "$DF" --data "$DATA" --extension $EXT --window 70 --save "$PARENT/dfc_cc200"