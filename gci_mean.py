import pathlib

from feature_extraction.granger import MatrixMean
import pandas as pd
import argparse


def main(df_path, data_dir, extension, binary):
    df = pd.read_csv(df_path)

    mean_make = MatrixMean(df, data_dir, extension, binary)
    mean_make.mean_gci()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--df', type=pathlib.Path)
    parser.add_argument('--data', type=pathlib.Path)
    parser.add_argument('--ext', type=str)
    parser.add_argument('--binary', help='binary classes or multiclass', action='store_true')
    parser.add_argument('--no-binary', dest='binary', action='store_false')

    args = parser.parse_args()

    main(df_path=args.df, data_dir=args.data, extension=args.ext, binary=args.binary)
