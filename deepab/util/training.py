import os

from deepab.preprocess.create_antibody_db import download_train_dataset


def check_for_h5_file(h5_file, antibody_to_h5, ab_dir):
    """Checks for a H5 file. If unavailable, downloads/creates files from SabDab."""
    if not os.path.isfile(h5_file):
        print('No H5 file found at {}, creating new file in {}/ ...'.format(
            h5_file, ab_dir))
        if not os.path.isdir(ab_dir):
            print('{}/ does not exist, creating {}/ ...'.format(
                ab_dir, ab_dir))
            os.mkdir(ab_dir)
        pdb_files = [f.endswith('pdb') for f in os.listdir(ab_dir)]
        if len(pdb_files) == 0:
            print('No PDB files found in {}, downloading PDBs ...'.format(
                ab_dir))
            download_train_dataset()
        print('Creating new h5 file at {} using data from {}/ ...'.format(
            h5_file, ab_dir))
        antibody_to_h5(ab_dir, h5_file, print_progress=True)
