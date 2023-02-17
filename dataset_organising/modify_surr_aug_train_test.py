import pathlib
import pandas as pd
import argparse
import os
import numpy as np


def remove_low(in_df, data_dir, save=True, save_path=0):
    all_files = os.listdir(data_dir)

    all_paths = [data_dir + file for file in all_files]

    ids = []
    for i, f in enumerate(all_paths):
        got = np.load(f)
        if len(got) < 116:
            ids.append(all_files[i])

    cut = [ts.replace('.npy', '_dfc.npy') for ts in ids]
    new_df = in_df[~in_df.FILE_ID.isin(cut)]

    if save:
        new_df.to_csv(save_path, index=False)


def main(data_dir, df_from_path=True, df_path=0, df=0, save=False, save_path=0):
    if df_from_path:
        df = pd.read_csv(df_path)
    else:
        df = df
    remove_low(in_df=df, data_dir=data_dir, save=save, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ModifySurrFile',
        description='test_surr.csv and train_surr.csv contains files that arent in actual dfc dataset, due to have '
                    'low temporal samples some timesieres wer not calculates intp dfc arrays, remove these files ')

    parser.add_argument('--data', help='full path to data', type=pathlib.Path, required=True)
    parser.add_argument('--from_path', help='does original df need to be uploaded?',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--df_path', help='full path to original df, optional', type=pathlib.Path, required=False)
    parser.add_argument('--df', help='original .csv pd dataframe', required=False)
    parser.add_argument('--save', help='save the updated df?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_path', help='full save path', type=pathlib.Path, required=False)

    args = parser.parse_args()

    main(args.data, df_from_path=args.from_path, df_path=args.df_path, df=args.df, save=args.save, save_path=args.save_path)