import pandas as pd
import os
import re

class AddFileNames:
    def __init__(self, main_dir, dataframe_folder, dataframe_name):
        '''

        :param dataframe_name: (pandas dataframe) dataframe containing phenotypic data
        :param dataframe_folder: (subdirectory containing phynotypic files)
        '''

        self.main_dir = main_dir
        self.pheno_dir = os.path.join(main_dir, dataframe_folder)
        self.dataframe_path = os.path.join(self.pheno_dir, dataframe_name)

        self.df = pd.read_csv(self.dataframe_path, index_col=[0])

    def update_df(self, new_df):
        '''

        :param new_df: (Pandas Dataframe) new dataframe that was processed without class methods
        :return:
        '''
        self.df = new_df

    def add_feature(self, feature_name, feature_data_dir_name, feature_file_ext):
        '''
        :param feature_name: desired name of feature column
        :param feature_data_dir_name: directory containing feature data files
        :param feature_file_ext: file extension of feature files eg. 'nii.gz'
        :return:
        '''

        feature_dir = os.path.join(self.main_dir, feature_data_dir_name)
        feature_files = next(os.walk(feature_dir))[2]
        feature_files = [feature_files[i] for i in range(len(feature_files)) if feature_file_ext in feature_files[i]]

        feature_file_indexed = list(range(len(feature_files)))
        for i in range(len(feature_files)):
            sub_id = re.search('5\d+\d', feature_files[i]).group()
            index_find = self.df[self.df['SUB_ID'] == int(sub_id)].index[0]
            feature_file_indexed[index_find] = feature_files[i]

        self.df[feature_name] = feature_file_indexed

        return self.df

