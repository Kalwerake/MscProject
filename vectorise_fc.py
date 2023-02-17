import pandas as pd
import numpy as np
import os
import argparse
import pathlib
'''
produce fc matrices and vectorise store in one np array for machine learning 
'''


def fc_make(roi_path):
    roi = pd.read_csv(roi_path, sep='\t', lineterminator='\n')
    return roi.corr(method='pearson').to_numpy()


def upper_tri_indexing(cov_matrix):
    matrix_length = cov_matrix.shape[0]
    r, c = np.triu_indices(matrix_length, 1)
    return cov_matrix[r, c]


def main(df_path, data_dir, save_path):

    df = pd.read_csv(df_path)
    ids = df.FILE_ID
    target = df.DX_GROUP
    filenames = [sample+'_rois_cc200.1D' for sample in ids]
    paths = [os.path.join(data_dir, file) for file in filenames]

    all_fc = [fc_make(path) for path in paths]

    vectorised = np.stack([upper_tri_indexing(fc) for fc in all_fc])
    out_df = np.c_[vectorised, target]
    np.save(save_path, out_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='VectoriseFC',
        description='vectorise all functional connectivty matrices and store in .npy')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to time series data directory', type=pathlib.Path)
    parser.add_argument('--save', help='save directory path', type=pathlib.Path)

    args = parser.parse_args()

    main(df_path=args.df, data_dir=args.data, save_path=args.save)

