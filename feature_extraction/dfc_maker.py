import os
import pandas as pd
from dfc_functions import DFC

parent_dir = os.path.dirname(os.getcwd())# parent directory
phenotype_path = os.path.join(parent_dir, 'phenotype_files/pheno_clean.csv') #path to csv containing data
phenotype_df = pd.read_csv(phenotype_path) # phenotype data
roi_path = os.path.join(parent_dir, 'rois_cc200') # path to subdirectory containing ROI timeseries data
pickle_path = os.path.join(parent_dir, 'dfc_cc200') # path to desired subdirectory for .pkl storage

pickle_party = DFC(phenotype_df, roi_path, pickle_path) # DFC class instance iniatlise

pickle_party.pickle_jar(22) # make .pkl files containing dFC matrices window length of 22
