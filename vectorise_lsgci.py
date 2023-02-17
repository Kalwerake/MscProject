import pandas as pd
import numpy as np
import os
import argparse
import pathlib
'''
Vectorise the large scale connectivty matrices and store in one np array for machine learning 
'''

def main(df_path,data_dir,save_path):

    df = pd.read_csv(df_path)
    ids = df.FILE_ID
    target = df.DX_GROUP
    filenames = [sample+'_gci.npy' for sample in ids]
    paths = [os.path.join(data_dir,file) for file in filenames]
    vectorised = np.stack([np.load(p).flatten() for p in paths])
    out_df = np.c_[vectorised, target]
    np.save(save_path, out_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='VectoriseGCI',
        description='vectorise all gci connectivty matrices and store in .npy')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to time series data directory', type=pathlib.Path)
    parser.add_argument('--save', help='save directory path', type=pathlib.Path)

    args = parser.parse_args()

    main(df_path=args.df, data_dir=args.data, save_path=args.save)

