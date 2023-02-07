from dfc_functions import PictureThis
import pandas as pd
import os
parent_dir = os.path.dirname(os.getcwd())
pheno_dir = os.path.join(parent_dir, 'phenotype_files')  # Directory storing phenotypic data csv
df_path = os.path.join(pheno_dir, 'pheno_clean.csv') # path to the phenotypic csv file
df = pd.read_csv(df_path) # obtain phenotypic data using pandas


pickle_files= df['DFC_DATA_STORE'] # pickle filenames containing DFC data

picture_module = PictureThis(parent_dir, 'dfc_cc200', 'dfc_cc200_figs')
# make instance of PictureThis module for creating correlation matrices.
# 'dfc_cc200 contains the DFC data and 'dfc_cc200_figs' is the subdirectory name for matrix storage

for pickle in pickle_files:

    picture_module.matrix_make(pickle)

# matrix_make() will take each pickle file produce matrices and store in dfc_cc200_figs under subdirectories.
