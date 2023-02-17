import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import os

from pyunicorn.timeseries.surrogates import Surrogates

main_dir = os.getcwd()
data_dir = main_dir+'/rois_cc200'
annot_df_path = main_dir + '/phenotype_files/pheno_nn.csv'
df = pd.read_csv(annot_df_path)
save_dir = main_dir+'/twin_augmented'
os.mkdir(save_dir)

sub_id = df.FILE_ID
roi_paths = [os.path.join(data_dir, file) for file in df.CC200]

for i, path in enumerate(roi_paths):
    real_save_path = os.path.join(save_dir, sub_id[i] + '_real.npy')
    data = pd.read_csv(path, sep='\t', lineterminator='\n')
    data = data.to_numpy()
    np.save(real_save_path, data)
    data = normalize(data).T

    t = Surrogates(data)
    for j in range(100):
        save_name = '{}_surr_{:03d}.npy'.format(sub_id[i], j+1)
        surr_save = os.path.join(save_dir, save_name)
        s = t.twin_surrogates(original_data=data, dimension=3, delay=2, threshold=0.02)
        s = s.T
        np.save(surr_save, s)

    t.clear_cache()



