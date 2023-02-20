import pathlib
from feature_extraction.granger import gci, large_scale_gci
from feature_extraction.fc import FetchROI
import os
import pandas as pd
import numpy as np
import argparse


def main(df_path, roi_dir, gci_dir, extension, large_scale):
    pheno_df = pd.read_csv(df_path)

    subjects = pheno_df.FILE_ID
    roi_files = [sub + extension for sub in subjects]

    if '.1D' in extension:
        fetch = FetchROI(roi_dir)

    try:
        os.mkdir(gci_dir)
    except FileExistsError:
        pass

    for i, file in enumerate(roi_files):
        matrix_name = subjects[i] + '_gci.npy'
        matrix_path = os.path.join(gci_dir, matrix_name)
        if '.1D' in extension:
            data = fetch.fetch_roi_avg_ts(file)
        else:
            data = np.load(os.path.join(gci_dir, file))

        if large_scale:
            gci_matrix = large_scale_gci(data)
        else:
            gci_matrix = gci(data)

        np.save(matrix_path, gci_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LargeScaleGranger',
        description='Calculate and stores large scale granger causality')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to time series data directory', type=pathlib.Path)
    parser.add_argument('--save', help='save directory path', type=pathlib.Path)
    parser.add_argument('--suffix', help='file name suffix after id', type=str)
    parser.add_argument('--large', help='calculate large scale index or not', action='store_true')
    parser.add_argument('--no-large', dest='large', action='store_false')

    args = parser.parse_args()

    main(df_path=args.df, roi_dir=args.data, gci_dir=args.save, extension=args.suffix, large_scale=args.large)
