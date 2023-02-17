from dfc_functions import PickPickle
import pandas as pd
import numpy as np
import os
import pickle
import argparse
import pathlib


def main(df_path, data_path, extension):
    data_dir = data_path
    df = pd.read_csv(df_path)

    asd, control = df[df['DX_GROUP'] == 1].FILE_ID, df[df['DX_GROUP'] == 2].FILE_ID

    asd_files, control_files = [ca + extension for ca in asd], [cc + extension for cc in control]

    if '.pkl' in extension:
        picker = PickPickle(data_dir)

        all_asd, all_control = [picker.get_pickle(asd_files[i]) for i in range(len(asd_files))], \
                               [picker.get_pickle(control_files[i]) for i in range(len(control_files))]

        asd_mean, control_mean = np.mean(np.array(all_asd), axis=0), np.mean(np.array(all_control), axis=0)

        asd_mean_path, control_mean_path = os.path.join(data_dir, 'asd_mean.pkl'), \
                                           os.path.join(data_dir, 'control_mean.pkl')

        with open(asd_mean_path, 'wb') as ca:
            pickle.dump(asd_mean, ca, protocol=pickle.HIGHEST_PROTOCOL)

        with open(control_mean_path, 'wb') as cc:
            pickle.dump(control_mean, cc, protocol=pickle.HIGHEST_PROTOCOL)

    elif '.npy' in extension:
        asd_paths, control_paths = [os.path.join(data_dir, asd_f) for asd_f in asd_files], \
                                   [os.path.join(data_dir, c_f) for c_f in control_files]
        all_asd, all_control = [np.load(a_path) for a_path in asd_paths], \
                               [np.load(c_path) for c_path in control_paths]

        asd_mean, control_mean = np.mean(np.array(all_asd), axis=0), np.mean(np.array(all_control), axis=0)

        asd_mean_path, control_mean_path = os.path.join(data_dir, 'asd_mean.npy'), \
                                            os.path.join(data_dir, 'control_mean.npy')

        np.save(asd_mean_path, asd_mean), np.save(control_mean_path, control_mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LargeScaleGranger',
        description='Calculate and stores large scale granger causality')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to time series data directory', type=pathlib.Path)
    parser.add_argument('--suffix', help='file name suffix after id', type=str)

    args = parser.parse_args()

    main(df_path=args.df,data_path=args.data, extension=args.suffix)
