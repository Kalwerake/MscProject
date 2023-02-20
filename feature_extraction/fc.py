import os
import pandas as pd
import numpy as np


class FetchROI:
    """
    load ROI time series data
    """

    def __init__(self, roi_path):
        """
        iniatialise object
        :param roi_path: full file path to roi timeseries data
        """
        self.roi_dir = roi_path

    def fetch_roi_avg_ts(self, roi_file):
        """
        :param roi_file: specific time series  data file name
        :return:
        """
        roi_path = os.path.join(self.roi_dir, roi_file)  # path to time series data
        output = pd.read_csv(roi_path, sep='\t', lineterminator='\n')  # load tap seperated file as pandas dataframe
        return output


class FC:
    """
    Calculating and storing  functional correlation data in .pkl format.

    Input folder should contain extracted time series data based on CC200 atlas.

    Data must be stored in BIDS format, with file extension `.1D`, tab seperated '\t' and line terminator \n

    needs:
    import os
    import pickle
    import pandas as pd """

    def __init__(self, df, roi_folder, extension, save_dir):
        """
        description_df: (pandas DataFrame)
            pandas dataframe containing phenotypic data,
            and extracted time series data file names under column 'CC200'.
        roi_folder: (path)
                    input folder path containing all time series data as subdirectory in main directory.
        pickle_folder:(path)
                     subdirectory folder name for storing pickle files containing dynamic correlation data.

        """
        self.df = df
        # Access roi data file names
        # subdirectory containing  roi data
        self.roi_folder = roi_folder
        self.save_dir = save_dir

        self.roi_ids = df.FILE_ID
        self.roi_files = [idx + extension for idx in self.roi_ids]
        # subdirectory path for dfc data pickle storage

        # all file paths for accessing ROI time series data
        self.roi_paths_all = [os.path.join(self.roi_folder, file) for file in self.roi_files]

        self.fc_file_all = [idx + '_fc.npy' for idx in self.roi_ids]  # paths to all .pkl files

        self.save_paths_all = [os.path.join(self.save_dir, file) for file in self.fc_file_all]

    def fc(self):
        """
        call pickle_jar() method for extraction of DFC data and storage in .pkl format, no arguments needed.
        """

        for i, roi_path in enumerate(self.roi_paths_all):  # take index and value of list roi_paths_all containing
            # path names for all time series data
            ts_df = pd.read_csv(roi_path, sep='\t', lineterminator='\n')
            fc = ts_df.corr(method='pearson')

            np.save(self.save_paths_all[i], fc)


def upper_tri_indexing(cov_matrix):
    matrix_length = cov_matrix.shape[0]
    r, c = np.triu_indices(matrix_length, 1)
    return cov_matrix[r, c]
