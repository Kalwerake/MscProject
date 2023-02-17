import pandas as pd
import os
import shutil

main_parent = os.path.dirname(os.getcwd())  # parent directory
pheno_dir = os.path.join(main_parent, 'phenotype_files')  # subdirectory containing phenotypic files

train_df = pd.read_csv(os.path.join(pheno_dir, 'train_df.csv'))  # training data reference csv
validation_df = pd.read_csv(os.path.join(pheno_dir, 'validation_df.csv'))
test_df = pd.read_csv(os.path.join(pheno_dir, 'test_df.csv'))  # test data

destination_parent = os.path.join(main_parent, 'dataset')  # parent subdirectory for data storage

# all subdirectories split into train and class subdirectories
sub_dir1 = os.path.join(destination_parent, 'train', 'class_1')  # class label 1 for training set
sub_dir2 = os.path.join(destination_parent, 'train', 'class_2')
sub_dir3 = os.path.join(destination_parent, 'validation', 'class_1')
sub_dir4 = os.path.join(destination_parent, 'validation', 'class_2')
sub_dir5 = os.path.join(destination_parent, 'test', 'class_1')
sub_dir6 = os.path.join(destination_parent, 'test', 'class_2')

dir_list = [sub_dir1, sub_dir2, sub_dir3, sub_dir4, sub_dir5, sub_dir6]  # make list of subdirectories

for d in dir_list:  # loop through subdirectory list and make subdirectories
    os.makedirs(d)

train_1 = train_df.loc[train_df['TARGET'] == 1, 'FILE_ID'].values  # File IDs for class 1 training data
train_2 = train_df.loc[train_df['TARGET'] == 2, 'FILE_ID'].values  # File IDs for class 2 training data
validation_1 = validation_df.loc[test_df['TARGET'] == 1, 'FILE_ID'].values  # File IDs for class 1 validation data
validation_2 = validation_df.loc[test_df['TARGET'] == 2, 'FILE_ID'].values  # File IDs for class 2 validation data
test_1 = test_df.loc[test_df['TARGET'] == 1, 'FILE_ID'].values  # File IDs for class 1 test data
test_2 = test_df.loc[test_df['TARGET'] == 2, 'FILE_ID'].values  # File IDs for class 2 test data

data = [train_1, train_2, validation_1, validation_2, test_1, test_2]  # list of dataset names


# function for moving DFC matrices from dfc_cc200_figs to subdirectories for training, validation and testing
def mover(data_list, destination_name):
    """
    :param data_list: list of file ids to move
    :param destination_name: corresponding destination for subject data in data_list
    :return: all folders will be moved to correct subdirectory
    """
    source_parent = os.path.join(main_parent, 'dfc_cc200_figs')  # subdirectory containing all matrices
    for dir_ in data_list:  # Take all FILE_IDs in dataset
        source = os.path.join(source_parent, dir_)  # combine source parent dfc_cc200_figs and file id
        destination = os.path.join(destination_name)  # destination subdirectory
        shutil.move(source, destination)  # move subdirectory indicated in pandas dataframe to destination


for i, val in enumerate(dir_list):  # dir_list contains all destination subdirectories
    mover(data[i], val)  # take each ID list and move subject data to correct subdirectory
