from dfc_functions import FetchROI
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import eigh, pinv
from statsmodels.tsa.api import VAR
import pickle

roi_dir = os.path.join(os.getcwd(), 'rois_aal')
pheno_path = os.path.join(os.getcwd(), 'phenotype_files', 'aal_nn.csv')

pheno_df = pd.read_csv(pheno_path)

subjects = pheno_df.FILE_ID
aal_files = [i + '_rois_aal.1D' for i in subjects]

fetch = FetchROI(roi_dir)


def large_scale_gci(df, file_name):
    _, rois = df.shape
    X = df.to_numpy()  #
    Xn = normalize(X)  # normalize X
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

    lsGCI = np.zeros((rois, rois))
    for i in range(rois):
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

        for j in range(rois):
            if j != i:
                GCI = np.log(np.var(E_minus[:, j]) / np.var(E_hat[:, j]))
                lsGCI[i, j] = max(GCI,0)  # from i to j

    return lsGCI


gci_dir = os.path.join(os.getcwd(), 'gci_aal')
os.mkdir(gci_dir)

for i, file in enumerate(aal_files):
    matrix_name = subjects[i] + '_gci_aal.pkl'
    matrix_path = os.path.join(gci_dir, matrix_name)
    data = fetch.fetch_roi_avg_ts(file)

    gci_matrix = large_scale_gci(data, file)
    with open(matrix_path, 'wb') as d:
        pickle.dump(gci_matrix, d, protocol=pickle.HIGHEST_PROTOCOL)


