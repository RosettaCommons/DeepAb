import h5py
import numpy as np
import argparse
from tqdm import tqdm
import os
from glob import glob
import pandas as pd
import json
import itertools

import deepab
from deepab.util.util import _aa_dict, letter_to_num

h_components = ["fwh1", "cdrh1", "fwh2", "cdrh2", "fwh3", "cdrh3", "fwh4"]
h_cdr_names = ["h1", "h2", "h3"]
l_components = ["fwl1", "cdrl1", "fwl2", "cdrl2", "fwl3", "cdrl3", "fwl4"]
l_cdr_names = ["l1", "l2", "l3"]


def to_dict(text):
    return json.loads(text.replace("\'", "\""))


def combine_anarci_components(anarci_dict, components):
    seq_list = list(
        itertools.chain.from_iterable(
            [list(anarci_dict[c].values()) for c in components]))
    seq = "".join(seq_list)

    return seq


def extract_seq_components(anarci_dict, seq_components, cdr_names):
    seq = combine_anarci_components(anarci_dict, seq_components)

    cdr_range_dict = {}
    cdr_seq_dict = {}
    for i in range(0, 3):
        cdr_range_dict[cdr_names[i]] = [
            len(
                combine_anarci_components(anarci_dict,
                                          seq_components[:2 * i + 1])),
            len(
                combine_anarci_components(anarci_dict,
                                          seq_components[:2 * i + 2])) - 1
        ]
        cdr_seq_dict[cdr_names[i]] = combine_anarci_components(
            anarci_dict, ["cdr" + cdr_names[i]])

    return seq, cdr_range_dict, cdr_seq_dict


def process_csv_data(csv_file, print_progress=True, verbose=False):
    rep_info = to_dict(
        np.genfromtxt(csv_file,
                      max_rows=1,
                      dtype=str,
                      delimiter="\t",
                      comments=None).item())
    info_dict = {
        "species": rep_info["Species"],
        "isotype": rep_info["Isotype"],
        "b_type": rep_info["BType"],
        "b_source": rep_info["BSource"],
        "disease": rep_info["Disease"],
        "vaccine": rep_info["Vaccine"],
    }

    col_names = pd.read_csv(csv_file, skiprows=1, nrows=1).columns
    max_rows = int(1e6)
    oas_df = pd.read_csv(csv_file,
                         skiprows=1,
                         names=col_names,
                         header=None,
                         usecols=[
                             'ANARCI_status_light', 'ANARCI_status_heavy',
                             'ANARCI_numbering_heavy', 'ANARCI_numbering_light'
                         ])
    oas_df = oas_df.query(
        "ANARCI_status_light == 'good' and ANARCI_status_heavy == 'good'")
    oas_df = oas_df[['ANARCI_numbering_heavy', 'ANARCI_numbering_light']]

    data_list = []
    for index, (anarci_h_data, anarci_l_data) in enumerate(oas_df.values):
        missing_component = False
        for c in h_components:
            if not c in to_dict(anarci_h_data):
                if verbose:
                    print("Missing heavy component in index {}: {}".format(
                        index, c))
                missing_component = True
        for c in l_components:
            if not c in to_dict(anarci_l_data):
                if verbose:
                    print("Missing light component in index {}: {}".format(
                        index, c))
                missing_component = True

        if missing_component:
            continue

        heavy_prim, h_cdr_range_dict, h_cdr_seq_dict = extract_seq_components(
            to_dict(anarci_h_data), h_components, h_cdr_names)
        light_prim, l_cdr_range_dict, l_cdr_seq_dict = extract_seq_components(
            to_dict(anarci_l_data), l_components, l_cdr_names)

        data_list.append({
            "heavy_data": (heavy_prim, h_cdr_range_dict, h_cdr_seq_dict),
            "light_data": (light_prim, l_cdr_range_dict, l_cdr_seq_dict),
            "metadata":
            info_dict
        })

    return data_list


def sequences_to_h5(oas_csv_dir,
                    out_file_path,
                    overwrite=False,
                    print_progress=False,
                    verbose=False):

    oas_csv_files = glob(os.path.join(oas_csv_dir, "*.csv"))
    data_list = []
    for oas_csv in tqdm(oas_csv_files):
        data_list.extend(
            process_csv_data(oas_csv, print_progress=False, verbose=verbose))

    num_seqs = len(data_list)
    max_h_len = 200
    max_l_len = 200

    if overwrite and os.path.isfile(out_file_path):
        os.remove(out_file_path)
    h5_out = h5py.File(out_file_path, 'w')
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
    species_set = h5_out.create_dataset('species', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())
    isotype_set = h5_out.create_dataset('isotype', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())
    b_type_set = h5_out.create_dataset('b_type', (num_seqs, ),
                                       compression='lzf',
                                       dtype=h5py.string_dtype())
    b_source_set = h5_out.create_dataset('b_source', (num_seqs, ),
                                         compression='lzf',
                                         dtype=h5py.string_dtype())
    disease_set = h5_out.create_dataset('disease', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())
    vaccine_set = h5_out.create_dataset('vaccine', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())

    for index, data_dict in tqdm(enumerate(data_list),
                                 disable=(not print_progress)):
        # Extract sequence from OAS data
        heavy_prim, h_cdr_range_dict, h_cdr_seq_dict = data_dict["heavy_data"]
        light_prim, l_cdr_range_dict, l_cdr_seq_dict = data_dict["light_data"]
        metadata = data_dict["metadata"]

        cdr_range_dict = {}
        cdr_range_dict.update(h_cdr_range_dict)
        cdr_range_dict.update(l_cdr_range_dict)

        h_len_set[index] = len(heavy_prim)
        l_len_set[index] = len(light_prim)

        h_prim_set[index, :len(heavy_prim)] = np.array(
            letter_to_num(heavy_prim, _aa_dict))
        l_prim_set[index, :len(light_prim)] = np.array(
            letter_to_num(light_prim, _aa_dict))

        for h_set, name in [(h1_set, 'h1'), (h2_set, 'h2'), (h3_set, 'h3'),
                            (l1_set, 'l1'), (l2_set, 'l2'), (l3_set, 'l3')]:
            h_set[index] = np.array(cdr_range_dict[name])

        species_set[index] = metadata["species"]
        isotype_set[index] = metadata["isotype"]
        b_type_set[index] = metadata["b_type"]
        b_source_set[index] = metadata["b_source"]
        disease_set[index] = metadata["disease"]
        vaccine_set[index] = metadata["vaccine"]


def cli():
    project_path = os.path.abspath(os.path.join(deepab.__file__, "../.."))
    data_path = os.path.join(project_path, "data")

    desc = 'Creates h5 files from all the truncated antibody PDB files in a directory'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'oas_csv_dir',
        type=str,
        help=
        'The directory containing antibody sequence CSV files downloaded from OAS'
    )
    parser.add_argument(
        '--out_file',
        type=str,
        default=os.path.join(data_path, 'abSeq.h5'),
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
                        default=True)

    args = parser.parse_args()
    oas_csv_dir = args.oas_csv_dir
    out_file = args.out_file
    overwrite = args.overwrite

    sequences_to_h5(oas_csv_dir,
                    out_file,
                    overwrite=overwrite,
                    print_progress=True)


if __name__ == '__main__':
    cli()
