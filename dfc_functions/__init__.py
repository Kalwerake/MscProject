import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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
    Calculating and storing dynamic functional correlation data in .pkl format.

    Input folder should contain extracted time series data based on CC200 atlas.

    Data must be stored in BIDS format, with file extension `.1D`, tab seperated '\t' and line terminator \n

    needs:
    import os
    import pickle
    import pandas as pd """

    def __init__(self, description_df, roi_folder, pickle_folder):
        """
        description_df: (pandas DataFrame)
            pandas dataframe containing phenotypic data,
            and extracted time series data file names under column 'CC200'.
        roi_folder: (path)
                    input folder path containing all time series data as subdirectory in main directory.
        pickle_folder:(path)
                     subdirectory folder name for storing pickle files containing dynamic correlation data.

        """
        self.parent_dir = os.path.dirname(os.getcwd())
        self.df = description_df
        # Access roi data file names
        self.roi_files = description_df['CC200']
        # subdirectory containing  roi data
        self.roi_repo = roi_folder
        # subdirectory path for dfc data pickle storage
        self.pickle_repo = pickle_folder
        # make subdirectory for pickle storage
        os.mkdir(self.pickle_repo)

        # all file paths for accessing ROI time series data
        self.roi_paths_all = [os.path.join(self.roi_repo, file) for file in self.roi_files]

        self.pickle_file_all = [ts.replace('_rois_cc200.1D', '_dfc.pkl') for ts in
                                self.roi_files]  # paths to all .pkl files

        self.pickle_paths_all = [os.path.join(self.pickle_repo, file) for file in self.pickle_file_all]

    def pickle_jar(self):
        """
        call pickle_jar() method for extraction of DFC data and storage in .pkl format, no arguments needed.
        """

        for i, roi_path in enumerate(self.roi_paths_all):  # take index and value of list roi_paths_all containing
            # path names for all time series data
            ts_df = pd.read_csv(roi_path, sep='\t', lineterminator='\n')

            fc = ts_df.corr(method='pearson')

            with open(self.pickle_paths_all[i], 'wb') as handle:
                pickle.dump(fc, handle, protocol=pickle.HIGHEST_PROTOCOL)


class DFC:
    """
    Calculating and storing dynamic functional correlation data in .pkl format.

    Input folder should contain extracted time series data based on CC200 atlas.

    Data must be stored in BIDS format, with file extension `.1D`, tab seperated '\t' and line terminator \n

    needs:
    import os
    import pickle
    import pandas as pd """

    def __init__(self, description_df, roi_folder, pickle_folder):
        """
        description_df: (pandas DataFrame)
            pandas dataframe containing phenotypic data,
            and extracted time series data file names under column 'CC200'.
        roi_folder: (path)
                    input folder path containing all time series data as subdirectory in main directory.
        pickle_folder:(path)
                     subdirectory folder name for storing pickle files containing dynamic correlation data.

        """
        self.parent_dir = os.path.dirname(os.getcwd())
        self.df = description_df
        # Access roi data file names
        self.roi_files = description_df['CC200']
        # subdirectory containing  roi data
        self.roi_repo = roi_folder
        # subdirectory path for dfc data pickle storage
        self.pickle_repo = pickle_folder
        # make subdirectory for pickle storage
        os.mkdir(self.pickle_repo)

        # all file paths for accessing ROI time series data
        self.roi_paths_all = [os.path.join(self.roi_repo, file) for file in self.roi_files]

        self.pickle_file_all = [ts.replace('_rois_cc200.1D', '_dfc.pkl') for ts in
                                self.roi_files]  # names for .pkl storage

        self.pickle_paths_all = [os.path.join(self.pickle_repo, file) for file in self.pickle_file_all]

    def dfc_calculator(self, time_series_data, window_length):
        """
        time_series_data: (pandas DataFrame)
                        Pandas dataframe containing time series data
        """

        dfc = {}  # dict object stores all calculated correlation data one subject at a time, key is time window number
        for i in range(len(time_series_data)):
            if (i + window_length) <= len(time_series_data):  # keep window within index
                dfc[i + 1] = time_series_data.iloc[i:i + window_length].corr(method='pearson')  # move window by 1 step
            else:
                break

        return dfc

    def pickle_jar(self, window_length):
        """
        call pickle_jar() method for extraction of DFC data and storage in .pkl format, no arguments needed.
        """

        for i, roi_path in enumerate(self.roi_paths_all):  # take index and value of list roi_paths_all containing
            # path names for all time series data

            ts_df_raw = pd.read_csv(roi_path, sep='\t', lineterminator='\n')  # fetch time series data
            if len(ts_df_raw) >= 116:  # exclude all data with less than 116 time points
                ts_df = ts_df_raw.iloc[5:116, :]  # exclude first 5 time points and use only upto 116
                ts_df.reset_index(drop=True, inplace=True)  # reset_index to 0 for easier indexing

            else:
                continue

            dfc_dict = self.dfc_calculator(ts_df, window_length)  # calculate dynamic correlation data

            with open(self.pickle_paths_all[i], 'wb') as handle:
                pickle.dump(dfc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class PickPickle:
    """
        obtain dict object containing DFC data for a subject spec
        pickle_folder: (str)
                name of subdirectory containing .pkl files """

    def __init__(self, pickle_folder):
        self.pickle_dir = pickle_folder

    def get_pickle(self, dict_name):
        """dict_name: (filename.pkl)
                    .pkl filename containing DFC data

        """
        pickle_path = os.path.join(self.pickle_dir, dict_name)

        with open(pickle_path, 'rb') as f:
            loaded_dict = pickle.load(f)

        return loaded_dict


class PictureThis:
    '''
    Needs pillow
    main_dir: main working directory or root directory
    pickle_folder: subdirectory containing .pkl files for DFC data
    fig_folder: desired subdirectory for storing correlation matrix jpg files
    '''

    def __init__(self, main_dir, pickle_folder, fig_folder):
        self.main_dir = main_dir  # main directory for project
        self.pickle_repo = os.path.join(main_dir, pickle_folder)  # subdirectory containing DFC data
        self.fig_repo = os.path.join(main_dir, fig_folder)  # subdirectory for correlation matrix storage

        os.mkdir(self.fig_repo)  # make subdirectory for correlation matrix storage fig_folder

    def get_pickle(self, pickle_file):
        """
        Get pickle file containing DFC data for one subjecy
        pickle_file: (filename.pkl)
                    .pkl filename containing DFC data
        """
        pickle_path = os.path.join(self.pickle_repo, pickle_file)

        with open(pickle_path, 'rb') as f:
            loaded_dict = pickle.load(f)

        return loaded_dict

    def matrix_make(self, pickle_file):
        '''
        :param pickle_file: name of pickle file
        :return: matrices stored in folder
        '''
        location_subject = pickle_file.replace('_dfc.pkl', '')  # subject id unique identifier
        subject_dump_repo = os.path.join(self.fig_repo,
                                         location_subject)  # folder for storing matrices for each subject
        os.mkdir(subject_dump_repo)  # make folder for subject
        subject_dfc_data = self.get_pickle(pickle_file)  # use get_pickle() function to load DFC data of subject

        for i, key in enumerate(subject_dfc_data):
            subject_file = location_subject + "_{:03d}.jpg".format(key)
            # "_{:03d}".format(key).jpg for pytorch dataset class
            subject_fig_path = os.path.join(subject_dump_repo, subject_file)  # save path for each matrix

            plt.figure(figsize=(10, 10))
            sns.heatmap(subject_dfc_data[key], annot=False, center=0, xticklabels=False, yticklabels=False, cbar=False)
            plt.savefig(subject_fig_path, bbox_inches='tight', pad_inches=0)
            plt.close()


class TwinSurrDFC:
    """
    Calculating and storing dynamic functional correlation data in .pkl format.

    Input folder should contain extracted time series data based on CC200 atlas.

    Data must be stored in BIDS format, with file extension `.1D`, tab seperated '\t' and line terminator \n

    needs:
    import os
    import pickle
    import pandas as pd """

    def __init__(self, df, roi_folder, save_folder):
        """
        description_df: (pandas DataFrame)
            pandas dataframe containing phenotypic data,
            and extracted time series data file names under column 'CC200'.
        roi_folder: (path)
                    input folder path containing all time series data as subdirectory in main directory.
        pickle_folder:(path)
                     subdirectory folder name for storing pickle files containing dynamic correlation data.

        """
        self.parent_dir = os.path.dirname(os.getcwd())
        self.df = df
        # Access roi data file names
        self.sub_ids = df.FILE_ID
        surr_files = ['{}_surr_{:03d}.npy'.format(self.sub_ids[i], j+1) for i in range(len(self.sub_ids)) for j in range (100)]
        real_files = [f'{self.sub_ids[i]}_real.npy' for i in range(len(self.sub_ids))]
        self.roi_files = [*surr_files, *real_files]
        # subdirectory containing  roi data
        self.roi_dir = roi_folder
        # subdirectory path for dfc data pickle storage
        self.save_dir = save_folder
        # make subdirectory for pickle storage
        try:
            os.mkdir(self.save_dir)
        except FileExistsError:
            pass

        # all file paths for accessing ROI time series data
        self.roi_paths_all = [os.path.join(roi_folder, file) for file in self.roi_files]

        self.save_file_all = [self.roi_files[i].replace('.npy', '_dfc.npy') for i in range(len(self.roi_files))]

        self.save_paths_all = [os.path.join(self.save_dir, file) for file in self.save_file_all]

    @staticmethod
    def dfc_calculator(time_series_data, window_length):
        """
        time_series_data: (pandas DataFrame)
                        Pandas dataframe containing time series data
        """

        dfc = [] # save co-variance 2d arrays to list
        for i in range(len(time_series_data)):
            if (i + window_length) <= len(time_series_data):  # keep window within index
                dfc.append(np.corrcoef(time_series_data[i:i + window_length], rowvar=False))  # move window by 1 step
            else:
                break

        return np.stack(dfc) # return 3d array

    def pickle_jar(self, window_length):
        """
        call pickle_jar() method for extraction of DFC data and storage in .pkl format, no arguments needed.
        """

        for i, roi_path in enumerate(self.roi_paths_all):  # take index and value of list roi_paths_all containing
            # path names for all time series data

            ts_df_raw = np.load(roi_path)  # fetch time series data
            if len(ts_df_raw) >= 116:  # exclude all data with less than 116 time points
                ts_df = ts_df_raw[5:116]  # exclude first 5 time points and use only upto 116
            else:
                continue

            dfc_array = self.dfc_calculator(ts_df, window_length)  # calculate dynamic correlation data

            np.save(self.save_paths_all[i], dfc_array)




