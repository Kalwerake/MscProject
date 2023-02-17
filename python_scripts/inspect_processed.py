import pandas as pd
import numpy as np
import os
import argparse
import pathlib
"""
inspect shape of all MRI and remove rows in original dataframe containing  missing data. 
"""


def main(fail_path, df_path, data_dir, save_df_path):
    absent = np.loadtxt(fail_path, dtype=str)
    missing = [int(a[6:11]) for a in absent]
    df = pd.read_csv(df_path)
    sub_ids = df.SUB_ID

    n_missing = df.drop([i for i, sub in enumerate(sub_ids) if sub in missing])
    n_missing.reset_index(inplace=True,drop=True)
	
    processed = [os.path.join(data_dir, f'{sub}.npy') for sub in n_missing.SUB_ID]

    all_shapes = [[n_missing.SUB_ID[i], *np.load(file).shape] for i, file in enumerate(processed)]

    out_df = pd.DataFrame(all_shapes, columns=['SUB_ID', 'x', 'y', 'z'])

    n_missing.to_csv(save_df_path, index=False)
    out_df.to_csv(os.path.join(data_dir, 'shapes.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='GetShapes',
        description='get shapes of processed data to ensure regularity')

    parser.add_argument('--fail', help='path to description fail.txt', type=pathlib.Path)
    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to data directory', type=pathlib.Path)
    parser.add_argument('--save', help='path to save new phenotype file', type=pathlib.Path)

    args = parser.parse_args()

    main(fail_path=args.fail, df_path=args.df, data_dir=args.data, save_df_path=args.save)
