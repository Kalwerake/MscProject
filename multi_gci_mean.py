from dfc_functions import PickPickle
import pandas as pd
import numpy as np
import os
import pickle

aal_dir = os.path.join(os.getcwd(), 'gci_aal')
cc200_dir = os.path.join(os.getcwd(), 'gci_cc200')
pheno_dir = os.path.join(os.getcwd(), 'phenotype_files')
cc200_df_p = os.path.join(pheno_dir, 'pheno_nn.csv')
aal_df_p = os.path.join(pheno_dir, 'aal_nn.csv')

cc200_df = pd.read_csv(cc200_df_p)
aal_df = pd.read_csv(aal_df_p)


class AverageStore:
    """
    :param data_dir : directory containing matrices in pickle format
    :param df : phenotype file containing class labels and file ids
    """

    def __init__(self, data_dir, df):
        self.dir = data_dir
        self.df = df
        self.fetch = PickPickle(data_dir)  # inialise object for loading pickles

    def add_suffix(self, column_name, extension, label):
        """
        :param column_name: class label column name
        :param extension: file extension and unique identifier after file id
        :param label: class label needed for extraction
        :return:
        """
        file_ids = self.df[self.df[column_name] == label].FILE_ID
        return [ca + extension for ca in file_ids]

    def split_save(self, column_name, extension, **groups):
        """
        :param column_name: class label column name
        :param extension: file extension and unique identifier after file id
        :param groups: in form <class_name> = <class_label>
        :return:
        """
        for name, label in groups.items():  # parse **groups obtain names, labels
            files = self.add_suffix(column_name, extension, label)  # get file names for matrices
            all_matrices = [self.fetch.get_pickle(files[i]) for i in
                            range(len(files))]  # stack each matrix into 3d list
            mean = np.mean(np.array(all_matrices),
                           axis=0)  # make 3d matrix into 3d numpy array and find elementwise mean
            path = os.path.join(self.dir, f'{name}_mean.pkl')  # path for saving mean matrix
            with open(path, 'wb') as ca:  # pickle dump matrix to path
                pickle.dump(mean, ca, protocol=pickle.HIGHEST_PROTOCOL)


cc200_store = AverageStore(cc200_dir, cc200_df)
cc200_store.split_save('DX_GROUP', '_gci.pkl', asd=1, control=2)
cc200_store.split_save('DSM_IV_TR', '_gci.pkl', mc_0=0, mc_1=1, mc_2=2, mc_3=3, mc_4=4)

aal_store= AverageStore(aal_dir, aal_df)
aal_store.split_save('DX_GROUP', '_gci_aal.pkl', asd=1, control=2)
aal_store.split_save('DSM_IV_TR', '_gci_aal.pkl', mc_0=0, mc_1=1, mc_2=2, mc_3=3, mc_4=4)