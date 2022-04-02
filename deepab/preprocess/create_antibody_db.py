#!/usr/bin/env python
#
# (c) Copyright Rosetta Commons Member Institutions.
# (c) This file is part of the Rosetta software suite and is made available under license.
# (c) The Rosetta software is developed by the contributing members of the Rosetta Commons.
# (c) For more information, see http://www.rosettacommons.org. Questions about this can be
# (c) addressed to University of Washington CoMotion, email: license@uw.edu.

## @file   create_antibody_db.py
## @brief  Script to create database for RosettaAntibody
## @author Jeliazko Jeliazkov

## @details download non-redundant Chothia Abs from SAbDab
## Abs are downloaded by html query (is there a better practice)?
## Abs are Chothia-numbered, though we use Kabat to define CDRs.
## After download, trim Abs to Fv and extract FR and CDR sequences.
## For the purpose of trimming, truncate the heavy @112 and light @109
## Some directories (antibody_database, info, etc...) are hard coded.

import os
import sys
import requests
import pandas as pd
import argparse
from Bio import SeqIO
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import deepab
from deepab.util.pdb import get_pdb_chain_seq
from deepab.util.util import RawTextArgumentDefaultsHelpFormatter

# Relative paths from deepab/data
_DEFAULT_SUMMARY_FILE_PATH = 'info/sabdab_summary.tsv'
_DEFAULT_ANTIBODY_DATABASE_PATH = 'antibody_database/'
_TESTSET_FILENAME_ = 'TestSetList.txt'


def parse_sabdab_summary(file_name):
    """
    SAbDab produces a rather unique summary file.
    This function reads that file into a dict with the key being the
    4-letter PDB code.
    """
    # dict[pdb] = {col1 : value, col2: value, ...}
    sabdab_dict = {}

    with open(file_name, "r") as f:
        # first line is the header, or all the keys in our sub-dict
        header = f.readline().strip().split("\t")
        # next lines are data
        for line in f.readlines():
            split_line = line.strip().split("\t")
            td = {}  # temporary dict of key value pairs for one pdb
            for k, v in zip(header[1:], split_line[1:]):
                # pdb id is first, so we skip that for now
                td[k] = v
            # add temporary dict to sabdab dict at the pdb id
            sabdab_dict[split_line[0]] = td

    return sabdab_dict


def truncate_chain(pdb_text, chain, resnum, newchain):
    """
    Read PDB line by line and return all lines for a chain,
    with a resnum less than or equal to the input.
    This has to be permissive for insertion codes.
    This will return only a single truncated chain.
    This function can update chain to newchain.
    """
    trunc_text = ""
    for line in pdb_text.split("\n"):
        if (line.startswith("ATOM") and line[21] == chain
                and int(line[22:26]) <= resnum):
            trunc_text += line[:21] + newchain + line[22:]
            trunc_text += "\n"
    return trunc_text


def truncate_fasta_for_pdb(fasta_file, pdb_file, warn_pdbs, pdb_id=""):
    fasta_sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        s = record.seq._data
        if type(s) == bytes:
            s = s.decode()
        fasta_sequences.append(s)

    try:
        pdb_h_seq = get_pdb_chain_seq(pdb_file, "H")
        pdb_l_seq = get_pdb_chain_seq(pdb_file, "L")
    except:
        print("Missing chain for PDB ID: {}".format(pdb_id))
        warn_pdbs.append(pdb_id)
        return

    if pdb_h_seq == None or pdb_l_seq == None or len(pdb_h_seq) == 0 or len(
            pdb_l_seq) == 0:
        bad_h = pdb_h_seq == None or len(pdb_h_seq) == 0
        print("Missing {} chain seq for PDB: {}".format(
            "H" if bad_h else "L", pdb_id))
        warn_pdbs.append(pdb_id)
        return

    cterm_length = 15
    h_cterm_seq = pdb_h_seq[-cterm_length:]
    for s in fasta_sequences:
        if h_cterm_seq in s:
            trunc_h_seq = s[:s.index(h_cterm_seq) + cterm_length]

    l_cterm_seq = pdb_l_seq[-cterm_length:]
    for s in fasta_sequences:
        if l_cterm_seq in s:
            trunc_l_seq = s[:s.index(l_cterm_seq) + cterm_length]

    try:
        with open(fasta_file, "w") as f:
            f.write(">{0}:H\n{1}\n>{0}:L\n{2}\n\n".format(
                pdb_id, trunc_h_seq, trunc_l_seq))
    except:
        print("Failed to write truncated fasta for PDB ID {}".format(pdb_id))
        print("\tFasta file\t{}".format(fasta_file))
        print("\tPDB file\t{}".format(pdb_file))

        warn_pdbs.append(pdb_id)


def _get_HL_chains(pdb_path):
    """Gets the heavy and light chain ID's from a chothia file from SAbDab"""
    # Get the line with the HCHAIN and LCHAIN
    hl_line = ''
    with open(pdb_path) as f:
        for line in f.readlines():
            if 'PAIRED_HL' in line:
                hl_line = line
                break
    if hl_line == '':
        return None

    words = hl_line.split(' ')
    h_chain = l_chain = None
    for word in words:
        if word.startswith('HCHAIN'):
            h_chain = word.split('=')[1]
        if word.startswith('LCHAIN'):
            l_chain = word.split('=')[1]
    return h_chain, l_chain


def truncate_antibody_pdb(pdb_id, antibody_database_path, same_chain,
                          warn_pdbs, ignore_same_VL_VH_chains,
                          sabdab_summary_path):
    """
    :param pdb_id: The PDB ID of the protein
    :param antibody_database_path:
        The path where all the chothia numbered pdb files are stored as [pdb_id].pdb
    :param ignore_same_VL_VH_chains:
        Whether or not to ignore a PDB file when its VL chain and VH chains are
    :type ignore_same_VL_VH_chains: bool
    """
    # check if 'pdb_trunc.pdb' exists, if not then generate it
    pdb_path = os.path.join(antibody_database_path, pdb_id + '.pdb')
    trunc_pdb_path = os.path.join(
        os.path.join(antibody_database_path, pdb_id + '_trunc.pdb'))
    if os.path.isfile(trunc_pdb_path):
        print('We think a truncated version of ' + pdb_id +
              ' was found. Skipping.')
        os.remove(pdb_path)  # delete old file just in case
        return

    pdb_text = ''
    try:
        with open(pdb_path, 'r') as f:
            pdb_text = f.read()  # want string not list
    except IOError:
        sys.exit('Failed to open {} in antibody_database/ !'.format(pdb_id))

    # should have pdb_text now
    if len(pdb_text) == 0:
        sys.exit('Nothing parsed for PDB {} !'.format(pdb_id))

    # Get the hchain and lchain data from the SAbDab summary file, if givenj
    hchain_text, lchain_text = '', ''
    if sabdab_summary_path is not None:
        sabdab_dict = parse_sabdab_summary(sabdab_summary_path)
        hchain, lchain = sabdab_dict[pdb_id]["Hchain"], sabdab_dict[pdb_id][
            "Lchain"]
    else:
        hchain, lchain = _get_HL_chains(pdb_path)

    # we do not currently have a good way of handling VH & VL on the same chain
    if ignore_same_VL_VH_chains and hchain == lchain:
        same_chain.append(pdb_id)
        print(pdb_id + ' has the VH+VL on a single chain, removing...')
        os.remove(antibody_database_path + pdb_id + '.pdb')
        return

    if not hchain == 'NA':
        hchain_text = truncate_chain(pdb_text, hchain, 112, 'H')
        if len(hchain_text) == 0:
            # could not find heavy chain -- do not overwrite, but warn!
            warn_pdbs.append(pdb_id)
            print('Warning, could not find ' + hchain + ' chain for ' +
                  pdb_id + ' !')
            print(
                'It was not reported to be NA, so the file may have been altered!'
            )
            return

    if not lchain == 'NA':
        lchain_text = truncate_chain(pdb_text, lchain, 109, 'L')
        if len(lchain_text) == 0:
            # could not find heavy chain -- do not overwrite, but warn!
            warn_pdbs.append(pdb_id)
            print('Warning, could not find ' + lchain + ' chain for ' +
                  pdb_id + ' !')
            print(
                'It was not reported to be NA, so the file may have been altered!'
            )
            return

    # write new file to avoid bugs from multiple truncations
    with open(trunc_pdb_path, 'w') as f:
        f.write(hchain_text + lchain_text)

    # remove old file
    os.remove(antibody_database_path + pdb_id + '.pdb')


def truncate_antibody_pdbs(
        antibody_database_path=_DEFAULT_ANTIBODY_DATABASE_PATH,
        sabdab_summary_path=_DEFAULT_SUMMARY_FILE_PATH,
        ignore_same_VL_VH_chains=True):
    """
    We only use the Fv as a template, so this function loads each pdb
    and deletes excess chains/residues. We define the Fv, under the
    Chothia numbering scheme as H1-H112 and L1-L109.
    """
    warn_pdbs = []  # count warnings
    same_chain = []  # count deleted files (VH+VL same chain)

    # iterate over PDBs in antibody_database and truncate accordingly
    unique_pdbs = set([
        x[:4] for x in os.listdir(antibody_database_path) if x.endswith(".pdb")
    ])

    print('Truncating pdb files...')
    for pdb in tqdm(unique_pdbs):
        truncate_antibody_pdb(
            pdb,
            antibody_database_path,
            same_chain,
            warn_pdbs,
            sabdab_summary_path=sabdab_summary_path,
            ignore_same_VL_VH_chains=ignore_same_VL_VH_chains)

    for pdb in same_chain:
        print("Deleted " + pdb +
              " from database because it has VH+VL on the same chain.")
    print("Deleted {} total of same chain VH+VLs.".format(len(same_chain)))

    if len(warn_pdbs) > 0:
        print("Finished truncating, with {} warnings.".format(len(warn_pdbs)))

    return warn_pdbs


def truncate_fasta_files(pdb_ids,
                         antibody_database_path=_DEFAULT_ANTIBODY_DATABASE_PATH
                         ):
    warn_pdbs = []

    print('Truncating fasta files...')
    for pdb in tqdm(pdb_ids):
        pdb_file = os.path.join(antibody_database_path,
                                "{}_trunc.pdb".format(pdb))
        fasta_file = os.path.join(antibody_database_path,
                                  "{}.fasta".format(pdb))
        truncate_fasta_for_pdb(fasta_file, pdb_file, warn_pdbs, pdb_id=pdb)
        os.system("mv {} {}".format(
            fasta_file,
            os.path.join(antibody_database_path,
                         "{}_trunc.fasta".format(pdb))))

    if len(warn_pdbs) > 0:
        print("Finished truncating, with {} warnings.".format(len(warn_pdbs)))

    return warn_pdbs


def download_file(url, output_path):
    with open(output_path, 'w') as f:
        f.write(requests.get(url).content.decode('utf-8'))


def download_chothia_pdb_files(pdb_ids,
                               antibody_database_path,
                               max_workers=16):
    """
    :param pdb_ids: A set of PDB IDs to download
    :type pdb_ids: set(str)
    :param antibody_database_path: Path to the directory to save the PDB files to.
    :type antibody_database_path: str
    :param max_workers: Max number of workers in the thread pool while downloading.
    :type max_workers: int
    """
    # Get PDBs to download from the summary file and the path to save each to.
    pdb_file_paths = [
        os.path.join(antibody_database_path, pdb + '.pdb') for pdb in pdb_ids
    ]

    # Download PDBs using multiple threads
    download_url = 'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{}/?scheme=chothia'
    urls = [download_url.format(pdb) for pdb in pdb_ids]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [
            executor.submit(lambda a: download_file(*a), args)
            for args in zip(urls, pdb_file_paths)
        ]
        print('Downloading chothia files to {}/ from {} ...'.format(
            antibody_database_path, download_url))
        for _ in tqdm(as_completed(results), total=len(urls)):
            pass


def download_fasta_files(pdb_ids, antibody_database_path, max_workers=16):
    """
    :param pdb_ids: A set of PDB IDs to download
    :type pdb_ids: set(str)
    :param antibody_database_path: Path to the directory to save the fasta files to.
    :type antibody_database_path: str
    :param max_workers: Max number of workers in the thread pool while downloading.
    :type max_workers: int
    """
    # Get fastas to download from the summary file and the path to save each to.
    fasta_file_paths = [
        os.path.join(antibody_database_path, pdb + '.fasta') for pdb in pdb_ids
    ]

    # Download fastas using multiple threads
    download_url = 'https://www.rcsb.org/fasta/entry/{}'
    urls = [download_url.format(pdb) for pdb in pdb_ids]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [
            executor.submit(lambda a: download_file(*a), args)
            for args in zip(urls, fasta_file_paths)
        ]
        print('Downloading fasta files to {}/ from {} ...'.format(
            antibody_database_path, download_url))
        for _ in tqdm(as_completed(results), total=len(urls)):
            pass


def download_and_truncate_pdbs(pdb_ids, antibody_database_path,
                               sabdab_summary_path, ignore_same_VL_VH_chains):
    warn_pdbs = []

    download_chothia_pdb_files(pdb_ids=pdb_ids,
                               antibody_database_path=antibody_database_path)
    warn_pdbs += truncate_antibody_pdbs(
        antibody_database_path=antibody_database_path,
        sabdab_summary_path=sabdab_summary_path,
        ignore_same_VL_VH_chains=ignore_same_VL_VH_chains)
    download_fasta_files(pdb_ids,
                         antibody_database_path=antibody_database_path)
    warn_pdbs += truncate_fasta_files(
        pdb_ids, antibody_database_path=antibody_database_path)

    if len(warn_pdbs) > 0:
        print("Removing {} PDBs from antibody database".format(len(warn_pdbs)))
        for pdb in warn_pdbs:
            os.system("rm {}".format(
                os.path.join(antibody_database_path, "{}*".format(pdb))))


def download_sabdab_summary_file(summary_file_path=_DEFAULT_SUMMARY_FILE_PATH,
                                 seqid=99,
                                 paired=True,
                                 nr_complex='All',
                                 nr_rfactor='',
                                 nr_res=4):
    base_url = 'http://opig.stats.ox.ac.uk'
    search_url = base_url + '/webapps/newsabdab/sabdab/search/'
    params = dict(seqid=seqid,
                  paired=paired,
                  nr_complex=nr_complex,
                  nr_rfactor=nr_rfactor,
                  nr_res=nr_res)
    query = requests.get(search_url, params=params)
    html = BeautifulSoup(query.content, 'html.parser')
    summary_file_url = base_url + html.find(
        id='downloads').find('a').get('href')
    print('Downloading sabdab summary to {} from: {} ...'.format(
        summary_file_path, summary_file_url))

    os.makedirs(os.path.split(summary_file_path)[0], exist_ok=True)
    download_file(summary_file_url, summary_file_path)


def download_test_dataset(test_set_path='test_set/'):
    # Change the working directory to the deepab/data directory
    original_dir = os.getcwd()
    project_path = os.path.abspath(os.path.join(deepab.__file__, "../.."))
    data_dir = Path(os.path.join(project_path, "data"))
    os.chdir(data_dir)

    dir_ = test_set_path
    if os.path.isdir(dir_):
        print('Clearing {} directory...'.format(dir_))
        for file in os.listdir(dir_):
            os.remove(os.path.join(dir_, file))
    else:
        print('Making {} directory...'.format(dir_))
        os.mkdir(dir_)

    test_dataset = pd.read_csv(str(data_dir.joinpath(_TESTSET_FILENAME_)))
    pdb_ids = set(test_dataset['PDB_ID'].unique())

    download_and_truncate_pdbs(pdb_ids,
                               test_set_path,
                               sabdab_summary_path=None,
                               ignore_same_VL_VH_chains=False)

    os.chdir(original_dir)


def download_train_dataset(
        summary_file_path=_DEFAULT_SUMMARY_FILE_PATH,
        antibody_database_path=_DEFAULT_ANTIBODY_DATABASE_PATH,
        download_summary_file=True,
        paired=False,
        **kwargs):
    """
    Downloads a training set from SAbDab, avoids downloading PDB files in the
    deepab/data/TestSetList.txt file.

    :param summary_file_path: Path to the summary file produced by SAbDab
    :type summary_file_path: str
    :param antibody_database_path: Path to the directory to save the PDB files to.
    :type antibody_database_path: str
    :param max_workers: Max number of workers in the thread pool while downloading.
    :type max_workers: int
    """
    # Change the working directory to the deepab/data directory
    original_dir = os.getcwd()
    project_path = os.path.abspath(os.path.join(deepab.__file__, "../.."))
    data_dir = Path(os.path.join(project_path, "data"))
    os.chdir(data_dir)

    # Make required directories. If they already exist, delete all contents
    dir_ = antibody_database_path
    if os.path.isdir(dir_):
        print('Clearing {} directory...'.format(dir_))
        for file in os.listdir(dir_):
            os.remove(os.path.join(dir_, file))
    else:
        print('Making {} directory...'.format(dir_))
        os.mkdir(dir_)

    # Get all the PDB ID's for the training set
    if download_summary_file:
        download_sabdab_summary_file(summary_file_path=summary_file_path,
                                     **kwargs)
    summary_dataframe = pd.read_csv(summary_file_path, sep='\t')
    all_pdbs = set(summary_dataframe['pdb'].unique())

    # Remove PDB's that appear in the test set
    test_dataset = pd.read_csv(str(data_dir.joinpath(_TESTSET_FILENAME_)))
    test_dataset['PDB_ID'] = test_dataset['PDB_ID'].str.lower()
    training_set_ids = all_pdbs - set(test_dataset['PDB_ID'].unique())

    download_and_truncate_pdbs(training_set_ids,
                               antibody_database_path,
                               sabdab_summary_path=summary_file_path,
                               ignore_same_VL_VH_chains=paired)

    os.chdir(original_dir)  # Go back to original working dir


def _cli():
    desc = ('''
        Downloads chothia files from SAbDab
        ''')
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument('--seqid',
                        type=int,
                        default=99,
                        help='Max sequence identity (%)')
    parser.add_argument('--paired',
                        default=True,
                        action="store_true",
                        help='Paired VH/VL only?')
    parser.add_argument(
        '--nr_complex',
        type=str,
        default='All',
        help='In complex? "All", "Bound only" or "Unbound only"')
    parser.add_argument('--nr_rfactor',
                        type=str,
                        default='',
                        help='R-Factor cutoff')
    parser.add_argument('--nr_res',
                        type=int,
                        default=4,
                        help='Resolution cutoff')
    parser.add_argument(
        '--summary_file',
        default=None,
        help='Summary file to use (will download if not provided)')
    args = parser.parse_args()

    summary_file = _DEFAULT_SUMMARY_FILE_PATH if args.summary_file == None else args.summary_file
    download_summary_file = summary_file == None or not os.path.isfile(
        summary_file)
    download_train_dataset(download_summary_file=download_summary_file,
                           summary_file_path=summary_file,
                           seqid=args.seqid,
                           paired=args.paired,
                           nr_complex=args.nr_complex,
                           nr_rfactor=args.nr_rfactor,
                           nr_res=args.nr_res)


if __name__ == '__main__':
    _cli()
