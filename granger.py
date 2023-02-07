import pathlib

from dfc_functions import FetchROI
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import eigh, pinv
from statsmodels.tsa.api import VAR

import argparse


def large_scale_gci(df, is_pd=True):
    if is_pd:
        X = df.to_numpy()  #
        Xn = normalize(X)  # normalize X
    else:
        Xn = normalize(df)

    cov = np.cov(Xn, rowvar=False)  # construct covariance matrix of features, state that feature data is not in row
    eigval, eigvec = eigh(cov)  # eigenvalue decomposition, eval(eigenvalues), eigvec(eigenvectors) is a matrix of
    # eigen vectors
    idx = eigval.argsort()[::-1]  # get indices of sorted eigenvalues and reverse list to get descending order
    eigval = eigval[idx]  # eigen values sorted in descending order
    W = eigvec[:, idx]  # eigen vectors sorted with respect to eigenvalues giving projection matrix W
    W_c = W[:, :35]  # choose the first 35 eigen vectors
    Z = np.dot(Xn, W_c)  # project data on to 35 dimensional space

    mvar = VAR(Z)
    results = mvar.fit(2)  # fit model with maxlag of 2

    z = results.fittedvalues  # model predictions has only 194 rows due to the lag of 2, I will add first two rows from X_ld
    z_hat = np.concatenate((Z[:2, :], z), axis=0)  # concatenate first two rows of X_ld to z
    W_plus = pinv(W_c)  # get pseudo inverse of projection matrix
    E_hat = Xn - np.dot(z_hat, W_plus)

    lsGCI = np.zeros((200, 200))
    for i in range(200):
        lsGCI[i, i] = 1
        X_iMinus = np.delete(Xn, i, 1)  # remove ith column from X remove feature from HD space
        W_iMinus = np.delete(W_c, i, 0)  # Remove ith row from projection matrix W
        Z_minus = np.dot(X_iMinus, W_iMinus)  # project matrix without column i onto 35d space
        mvar_minus = VAR(Z_minus)  # initialise new model with Z_minus
        results = mvar_minus.fit(2)  # fit model with maxlag of 2
        z_minus_pred = results.fittedvalues  # model predictions has only 194 rows due to the lag of 2, I will add first two rows from Z_minus
        z_m_hat = np.concatenate((Z_minus[:2, :], z_minus_pred), axis=0)  # concatenate first two rows of X_ld to z

        W_m_plus = pinv(W_iMinus)  # get pseudo inverse of W_iMinus
        E_m = X_iMinus - np.dot(z_m_hat, W_m_plus)  # get error matrix of predictions without ith feature
        E_minus = np.insert(E_m, i, 0, axis=1)  # add a dummy column at removed column

        for j in range(200):
            if j != i:
                GCI = np.log(np.var(E_minus[:, j]) / np.var(E_hat[:, j]))
                lsGCI[i, j] = max(GCI, 0)  # from i to j
            else:
                continue

    return lsGCI


def main(df_path, roi_dir, gci_dir, extension):
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

        gci_matrix = large_scale_gci(data)
        np.save(matrix_path, gci_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LargeScaleGranger',
        description='Calculate and stores large scale granger causality')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to time series data directory', type=pathlib.Path)
    parser.add_argument('--save', help='save directory path', type=pathlib.Path)
    parser.add_argument('--suffix', help='file name suffix after id', type=str)

    args = parser.parse_args()

    main(df_path=args.df, roi_dir=args.data, gci_dir=args.save, extension=args.save)
