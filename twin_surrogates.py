import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import os
import argparse
from pyunicorn.timeseries.surrogates import Surrogates
import pathlib

def main(df_path,data_dir,save_dir):
    df = pd.read_csv(df_path)
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='TwinSurrogates',
        description='Uses the twin surrogate method to synthesise multivariate time series data')
    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to data directory', type=pathlib.Path)
    parser.add_argument('--save', help='path to save directory', type=pathlib.Path)

    args = parser.parse_args()

    main(df_path=args.df, data_dir=args.data, save_dir=args.save)


