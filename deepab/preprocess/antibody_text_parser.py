import numpy as np
import pandas as pd
from os import listdir
from os.path import join
from os.path import basename, splitext
from Bio import SeqIO
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser

from deepab.util.pdb import protein_pairwise_geometry_matrix
from deepab.util.util import _aa_dict, letter_to_num


def get_id(pdb_file_path):
    return splitext(basename(pdb_file_path))[0]


def get_pdb_atoms(pdb_file_path):
    """Returns a list of the atom coordinates, and their properties in a pdb file
    :param pdb_file_path:
    :return:
    """
    with open(pdb_file_path, 'r') as f:
        lines = [line for line in f.readlines() if 'ATOM' in line]
    column_names = [
        'atom_num', 'atom_name', 'alternate_location_indicator',
        'residue_name', 'chain_id', 'residue_num',
        'code_for_insertions_of_residues', 'x', 'y', 'z', 'occupancy',
        'temperature_factor', 'segment_identifier', 'element_symbol'
    ]
    # Get the index at which each column starts/ends
    column_ends = np.array(
        [3, 10, 15, 16, 19, 21, 25, 26, 37, 45, 53, 59, 65, 75, 77])
    column_starts = column_ends[:-1] + 1
    column_ends = column_ends[1:]  # Ignore the first column (just says 'ATOM')

    rows = [[
        l[start:end + 1].replace(' ', '')
        for start, end in zip(column_starts, column_ends)
    ] for l in lines]
    return pd.DataFrame(rows, columns=column_names)


def antibody_db_seq_info(fasta_dir):
    fasta_files = [
        join(fasta_dir, _) for _ in listdir(fasta_dir) if _[-5:] == 'fasta'
    ]

    num_seqs = len(fasta_files)
    min_heavy_seq_len = min_light_seq_len = min_total_seq_len = float('inf')
    max_heavy_seq_len = max_light_seq_len = max_total_seq_len = -float('inf')

    for fasta_file in fasta_files:
        chains = list(SeqIO.parse(fasta_file, 'fasta'))
        if len(chains) != 2:
            msg = 'Expected 2 chains in {}, got {}'.format(
                fasta_file, len(chains))
            raise ValueError(msg)

        h_len = l_len = 0
        for chain in chains:
            if ':H' in chain.id or 'heavy' in chain.id:
                h_len = len(chain.seq)
                max_heavy_seq_len = max(h_len, max_heavy_seq_len)
                min_heavy_seq_len = min(h_len, min_heavy_seq_len)
            elif ':L' in chain.id or 'light' in chain.id:
                l_len = len(chain.seq)
                max_light_seq_len = max(l_len, max_light_seq_len)
                min_light_seq_len = min(l_len, min_light_seq_len)
            else:
                try:
                    chain_id = str(chain.id).split(':')[1]
                    msg = (
                        'Expected a heavy chain or light chain, marked as \'H\' '
                        ' or \'L\'. Got a chain id of :{} from protein {}')
                    raise ValueError(msg.format(chain_id, chain.id))
                except Exception:
                    raise ValueError(
                        '{} does not have >name:chain format'.format(
                            fasta_file))

        total_len = h_len + l_len
        max_total_seq_len = max(total_len, max_total_seq_len)
        min_total_seq_len = min(total_len, min_total_seq_len)

    return dict(num_seqs=num_seqs,
                max_heavy_seq_len=max_heavy_seq_len,
                min_heavy_seq_len=min_heavy_seq_len,
                max_light_seq_len=max_light_seq_len,
                min_light_seq_len=min_light_seq_len,
                max_total_seq_len=max_total_seq_len,
                min_total_seq_len=min_total_seq_len)


def get_chain_seqs(fasta_file_path):
    """Gets the sequnce of each chain in a fasta file
    :param fasta_file_path: The fasta file to read in.
    :return:
        A dictionary where the key is the chain id and the value is a list of
        ints corresponding to their amino acid.
    :rtype: dict
    """
    seqs = dict()
    for chain in SeqIO.parse(fasta_file_path, 'fasta'):
        if ':H' in chain.id or 'heavy' in chain.id:
            id_ = 'H'
        elif ':L' in chain.id or 'light' in chain.id:
            id_ = 'L'
        else:
            chain_id = str(chain.id).split(':')[1]
            msg = ('Expected a heavy chain or light chain, marked as \'H\' '
                   ' or \'L\'. Got a chain id of :{} from protein {}')
            raise ValueError(msg.format(chain_id, chain.id))

        seqs.update({id_: letter_to_num(str(chain.seq), _aa_dict)})
    return seqs


def get_cdr_indices(pdb_file_path):
    """Gets the indices of the CDR loop residues in the PDB file

    :param pdb_file_path: The pdb file to read.
    :return:
        A dictionary where the key is the loop name (h1, h2, h3, l1, l2, l3)
        and the value is a 2-tuple of the index range of residues in the loop.
    :rtype: dict
    """
    cdr_ranges = {
        'h1': [26, 35],
        'h2': [50, 65],
        'h3': [95, 102],
        'l1': [24, 34],
        'l2': [50, 56],
        'l3': [89, 97]
    }

    # Remove duplicate chainIDs (i.e. remove all ATOMS except for the first of
    # each chain) and reindex at 0
    data = get_pdb_atoms(pdb_file_path)
    data = data.drop_duplicates(
        ['chain_id', 'residue_num',
         'code_for_insertions_of_residues']).reset_index()

    # Get the 3 letter residue and residue ID for all the residues in the heavy chain
    heavy_chain_residues = data[data.chain_id == 'H']
    light_chain_residues = data[data.chain_id == 'L']

    # Extract the residues within the h3_cdr_range
    heavy_residue_nums = heavy_chain_residues.residue_num.astype('int32')
    h1_idxs = list(heavy_chain_residues[heavy_residue_nums.isin(
        cdr_ranges['h1'])].index)
    h2_idxs = list(heavy_chain_residues[heavy_residue_nums.isin(
        cdr_ranges['h2'])].index)
    h3_idxs = list(heavy_chain_residues[heavy_residue_nums.isin(
        cdr_ranges['h3'])].index)

    light_residue_nums = light_chain_residues.residue_num.astype('int32')
    l1_idxs = list(light_chain_residues[light_residue_nums.isin(
        cdr_ranges['l1'])].index)
    l2_idxs = list(light_chain_residues[light_residue_nums.isin(
        cdr_ranges['l2'])].index)
    l3_idxs = list(light_chain_residues[light_residue_nums.isin(
        cdr_ranges['l3'])].index)

    return dict(h1=h1_idxs,
                h2=h2_idxs,
                h3=h3_idxs,
                l1=l1_idxs,
                l2=l2_idxs,
                l3=l3_idxs)


def get_info(pdb_file, fasta_file=None, verbose=True, convert_to_degree=True):
    if fasta_file is not None:
        chain_seqs = get_chain_seqs(fasta_file)
    else:
        if verbose:
            print(
                'WARNING: No fasta file given to get_info(), getting sequence '
                'from PDB file')
        chain_seqs = dict()
        parser = PDBParser()
        structure = parser.get_structure(get_id(pdb_file), pdb_file)
        for chain in structure.get_chains():
            id_ = chain.id
            seq = seq1(''.join([residue.resname for residue in chain]))
            if id_ not in ['H', 'L']:
                msg = (
                    'Expected a heavy chain or light chain, marked as \'H\' '
                    ' or \'L\'. Got a chain id of :{} from protein {}')
                raise ValueError(msg.format(id_, get_id(pdb_file)))

            chain_seqs.update({id_: letter_to_num(seq, _aa_dict)})

    id_ = get_id(pdb_file)
    cdr_indices = get_cdr_indices(pdb_file)
    pairwise_geometry_mat = protein_pairwise_geometry_matrix(
        pdb_file, fasta_file=fasta_file, convert_to_degree=convert_to_degree)

    info = cdr_indices
    info.update(chain_seqs)
    info.update(dict(pairwise_geometry_mat=pairwise_geometry_mat, id=id_))
    return info
