import h5py
import numpy as np
import argparse
from tqdm import tqdm
from os import listdir, remove
import os.path as path
from pathlib import Path

from deepab.preprocess import antibody_text_parser as ab_parser


def antibody_to_h5(pdb_dir,
                   out_file_path,
                   fasta_dir=None,
                   overwrite=False,
                   print_progress=False):
    pdb_files = [_ for _ in listdir(pdb_dir) if _[-3:] == 'pdb']
    num_seqs = len(pdb_files)

    if fasta_dir is not None:
        seq_info = ab_parser.antibody_db_seq_info(fasta_dir)
        max_h_len = seq_info['max_heavy_seq_len']
        max_l_len = seq_info['max_light_seq_len']
        max_total_len = seq_info['max_total_seq_len']
    else:
        print('WARNING: No fasta directory given! Defaulting max sequence '
              'length for both heavy/lights chains to 300')
        max_h_len = 300
        max_l_len = 300
        max_total_len = 600

    if overwrite and path.isfile(out_file_path):
        remove(out_file_path)
    h5_out = h5py.File(out_file_path, 'w')
    id_set = h5_out.create_dataset('id', (num_seqs, ),
                                   compression='lzf',
                                   dtype='S25',
                                   maxshape=(None, ))
    h_len_set = h5_out.create_dataset('heavy_chain_seq_len', (num_seqs, ),
                                      compression='lzf',
                                      dtype='uint16',
                                      maxshape=(None, ),
                                      fillvalue=0)
    l_len_set = h5_out.create_dataset('light_chain_seq_len', (num_seqs, ),
                                      compression='lzf',
                                      dtype='uint16',
                                      maxshape=(None, ),
                                      fillvalue=0)
    h_prim_set = h5_out.create_dataset('heavy_chain_primary',
                                       (num_seqs, max_h_len),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, max_h_len),
                                       fillvalue=-1)
    l_prim_set = h5_out.create_dataset('light_chain_primary',
                                       (num_seqs, max_l_len),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, max_l_len),
                                       fillvalue=-1)
    h1_set = h5_out.create_dataset('h1_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    h2_set = h5_out.create_dataset('h2_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    h3_set = h5_out.create_dataset('h3_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    l1_set = h5_out.create_dataset('l1_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    l2_set = h5_out.create_dataset('l2_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    l3_set = h5_out.create_dataset('l3_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    pairwise_geometry_set = h5_out.create_dataset(
        'pairwise_geometry_mat', (num_seqs, 6, max_total_len, max_total_len),
        maxshape=(None, 6, max_total_len, max_total_len),
        compression='lzf',
        dtype='float',
        fillvalue=-1)

    for index, file in tqdm(enumerate(pdb_files),
                            disable=(not print_progress),
                            total=len(pdb_files)):
        # Get all file names
        id_ = ab_parser.get_id(file)
        pdb_file = str(path.join(pdb_dir, id_ + '.pdb'))

        fasta_file = None if fasta_dir is None else str(
            path.join(fasta_dir, id_ + '.fasta'))
        info = ab_parser.get_info(pdb_file,
                                  fasta_file=fasta_file,
                                  verbose=False)

        # Get primary structures
        heavy_prim = info['H']
        light_prim = info['L']

        total_len = len(heavy_prim) + len(light_prim)

        id_set[index] = np.string_(id_)

        try:
            pairwise_geometry_set[
                index, :6, :total_len, :total_len] = np.array(
                    info['pairwise_geometry_mat'])
        except TypeError:
            msg = ('Fasta/PDB coordinate length mismatch: the fasta sequence '
                   'length of {} and the number of coordinates ({}) in {} '
                   'mismatch.\n ')
            raise ValueError(
                msg.format(total_len, len(info['pairwise_geometry_mat']),
                           pdb_file))

        h_len_set[index] = len(heavy_prim)
        l_len_set[index] = len(light_prim)

        h_prim_set[index, :len(heavy_prim)] = np.array(heavy_prim)
        l_prim_set[index, :len(light_prim)] = np.array(light_prim)

        for h_set, name in [(h1_set, 'h1'), (h2_set, 'h2'), (h3_set, 'h3'),
                            (l1_set, 'l1'), (l2_set, 'l2'), (l3_set, 'l3')]:
            if len(info[name]) == 1:
                info[name] = [info[name][0], info[name][0]]
            # Skip loops that do not have residues
            if len(info[name]) == 2:
                h_set[index] = np.array(info[name])
            else:
                print(info[name])
                msg = 'WARNING: {} does not have any coordinates for the {} loop!'
                print(msg.format(file, name))


def cli():
    desc = 'Creates h5 files from all the truncated antibody PDB files in a directory'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'pdb_dir',
        type=str,
        help='The directory containing PDB files where an antibody'
        'with a PDB id of ID is named: ID.pdb')
    data_path = str(
        Path(path.dirname(path.realpath(__file__))).parent.joinpath('data'))
    data_path = path.join(data_path, 'antibody.h5')
    parser.add_argument(
        '--out_file',
        type=str,
        default=data_path,
        help='The name of the outputted h5 file. This should be a '
        'absolute path, otherwise it is output to the '
        'working directory.')
    parser.add_argument('--fasta_dir',
                        type=str,
                        default=None,
                        help='The directory containing fastas files where an '
                        'antibody with a PDB id of ID is named: ID.fasta')
    parser.add_argument('--overwrite',
                        action="store_true",
                        help='Whether or not to overwrite a file or not,'
                        ' if it exists',
                        default=False)

    args = parser.parse_args()
    pdb_dir = args.pdb_dir
    fasta_dir = args.fasta_dir
    out_file = args.out_file
    overwrite = args.overwrite

    antibody_to_h5(pdb_dir,
                   out_file,
                   fasta_dir=fasta_dir,
                   overwrite=overwrite,
                   print_progress=True)


if __name__ == '__main__':
    cli()
