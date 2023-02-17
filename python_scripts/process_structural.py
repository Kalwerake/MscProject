import numpy as np
import nibabel as nib
import pandas as pd
import argparse
import pathlib
import os


def main(df_path, data_dir, save_dir):
    df = pd.read_csv(df_path)
    file_list = [[f'sub-00{sub}_desc-brain_mask.nii.gz', f'sub-00{sub}_desc-preproc_T1w.nii.gz'] for sub in
                 df.SUB_ID]
    not_exist = []
    for m, s in file_list:
        try:
            m_img = nib.load(os.path.join(data_dir, m))

            f_img = nib.load(os.path.join(data_dir, s))
        except FileNotFoundError:
            not_exist.append(m)
            pass

        mask = m_img.get_fdata()
        img = f_img.get_fdata()

        im = mask * img
        scaled = (im - np.min(im)) / (np.max(im) - np.min(im))

        save_path = os.path.join(save_dir, f'{m[6:11]}.npy')

        np.save(save_path, scaled)

    fail_path = os.path.join(save_dir, 'fail.txt')

    with open(fail_path, 'a') as f:
        f.write('\n'.join(not_exist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PreprocessMRI',
        description='apply mask to data and normalise')

    parser.add_argument('--df', help='path to description csv', type=pathlib.Path)
    parser.add_argument('--data', help='path to data directory', type=pathlib.Path)
    parser.add_argument('--save', help='save directory path', type=pathlib.Path)

    args = parser.parse_args()

    main(df_path=args.df, data_dir=args.data, save_dir=args.save)
