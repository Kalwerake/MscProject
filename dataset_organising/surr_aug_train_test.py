import pathlib
import pandas as pd
import argparse


'''The original data annotation files once split into train and test sets contain unique identifiers in column FILE_ID, 
in order to keep train and test data separate, twin surrogates obtained from original data in train and test must be 
kept separate. This script will preserve segregation, by adding suffixes to original unique identifiers and create 
train and test annotation files that point towards augmented data.
surr_file_add function can be adopted for use on cross validation frameworks as well, simply use as package and import
'''


def surr_file_add(in_df, data_dir, save=False, save_path=0):
    '''
    :param data_dir:
    :param in_df: input original annotations pandas dataframe
    :param save: bool, if saving is required default False
    :param save_path: if save is required, input full path to save
    :return:
    '''

    sub_ids = in_df.FILE_ID  # file ids unique identifiers
    targets = in_df.TARGET  # targets
    surr_files = [['{}_surr_{:03d}_dfc.npy'.format(sub_ids[i], j + 1), targets[i]] for i in range(len(sub_ids)) for j in
                  range(100)]  # add suffix to ids indicating surrogate origin, add target to second column
    real_files = [[f'{sub_ids[i]}_real_dfc.npy', targets[i]] for i in
                  range(len(sub_ids))]  # add suffix indicating real origin
    roi_files = [*surr_files, *real_files]  # concatenate two nested lists
    out_df = pd.DataFrame(roi_files, columns=['FILE_ID', 'TARGET'])  # make pandas dataframe with column names
    if save:  # if save = True then save to save path as csv
        out_df.to_csv(save_path, index=False)
    else:
        return out_df



def main(df_from_path=True, df_path=0, df=0, save=False, save_path=0):
    if df_from_path:
        df = pd.read_csv(df_path)
    else:
        df = df
    surr_file_add(in_df=df, save=save, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='AddAugmentedFile',
        description='add surrogate augmented filenames to train test split or CV data files ')

    parser.add_argument('--from_path', help='does original df need to be uploaded?',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--df_path', help='full path to original df, optional', type=pathlib.Path, required=False)
    parser.add_argument('--df', help='original .csv pd dataframe', required=False)
    parser.add_argument('--save', help='save the updated df?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_path', help='full save path', type=pathlib.Path, required=False)

    args = parser.parse_args()

    main(df_from_path=args.from_path, df_path=args.df_path, df=args.df, save=args.save, save_path=args.save_path)
