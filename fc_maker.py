import pandas as pd
import os
import argparse
import pathlib
from feature_extraction.fc import FC


def main(df_path, data, extension, save):
    df = pd.read_csv(df_path)
    try:
        os.mkdir(save)
    except FileExistsError:
        pass

    fc_object = FC(df=df, roi_folder=data, extension=extension, save_dir=save)
    fc_object.fc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CalculateFC',
        description='get all FC matrices .npy')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to time series data directory', type=pathlib.Path)
    parser.add_argument('--extension', help='file extension for time series', type=str)
    parser.add_argument('--save', help='save directory path', type=pathlib.Path)

    args = parser.parse_args()

    main(df_path=args.df, data=args.data, extension=args.extension, save=args.save)

