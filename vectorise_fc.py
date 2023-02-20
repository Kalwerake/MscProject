import pandas as pd
import numpy as np
import os
import argparse
import pathlib
from feature_extraction.fc import upper_tri_indexing
'''
Vectorise all 2d matrices for machine learning
'''


def main(df_path, data_dir, ext, save_dir):

    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass

    df = pd.read_csv(df_path)
    ids = df.FILE_ID
    binary_target = [1 if target==1 else 0 for target in df.DX_GROUP]
    multi_target = df.DSM_IV_TR

    filenames = [sample + ext for sample in ids]
    paths = [os.path.join(data_dir, file) for file in filenames]

    if 'fc' in ext:
        all_fc = [np.load(path) for path in paths]
        vectorised = np.stack([upper_tri_indexing(fc) for fc in all_fc])
        vecs = np.c_[vectorised, binary_target, multi_target]
        out_df = pd.DataFrame(vecs)
        feature_number = vectorised.shape[1]

    elif 'gci' in ext:
        vectorised = np.stack([np.load(p).flatten() for p in paths])
        vecs = np.c_[vectorised, binary_target, multi_target]
        out_df = pd.DataFrame(vecs)
        diagonals = [i for i in range(out_df.shape[1]) if
                     len(out_df.iloc[:, i].unique()) == 1]  # get indexes of columns containing diagonal values
        out_df.drop(diagonals, axis=1, inplace=True)  # drop columns containing diagonal data

        feature_number = vectorised.shape[1] - len(diagonals)

    col_names = [f'{i}' for i in range(feature_number)]
    col_names.extend(['DX_GROUP', 'DSM_IV_TR'])
    out_df.columns = col_names

    out_df = out_df.astype({'DX_GROUP': 'int32', 'DSM_IV_TR': 'int32'})

    out_name = os.path.basename(data_dir) + '_vectorised.csv.gz'
    save_path = os.path.join(save_dir, out_name)
    out_df.to_csv(save_path, index=False, sep='\t', compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='VectoriseFC',
        description='vectorise all functional connectivity matrices and store in as compressed tsv')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to time series data directory', type=pathlib.Path)
    parser.add_argument('--ext', help='path to time series data directory', type=str)
    parser.add_argument('--save', help='save directory path', type=pathlib.Path)

    args = parser.parse_args()

    main(df_path=args.df, data_dir=args.data, ext=args.ext, save_dir=args.save)


