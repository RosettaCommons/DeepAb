import torch
import torch.nn.functional as F
import re
import argparse
from Bio import SeqIO


class RawTextArgumentDefaultsHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """CLI help formatter that includes the default value in the help dialog
    and formats as raw text i.e. can use escape characters."""
    pass


_aa_dict = {
    'A': '0',
    'C': '1',
    'D': '2',
    'E': '3',
    'F': '4',
    'G': '5',
    'H': '6',
    'I': '7',
    'K': '8',
    'L': '9',
    'M': '10',
    'N': '11',
    'P': '12',
    'Q': '13',
    'R': '14',
    'S': '15',
    'T': '16',
    'V': '17',
    'W': '18',
    'Y': '19'
}

_aa_1_3_dict = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
    '-': 'GAP'
}

_rev_aa_dict = {v: k for k, v in _aa_dict.items()}


def letter_to_num(string, dict_):
    """Function taken from ProteinNet (https://github.com/aqlaboratory/proteinnet/blob/master/code/text_parser.py).
    Convert string of letters to list of ints"""
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num


def one_hot_seq(seq):
    """Gets a one-hot encoded version of a protein sequence"""
    return F.one_hot(torch.LongTensor(letter_to_num(seq, _aa_dict)),
                     num_classes=20)


def load_full_seq(fasta_file):
    """Concatenates the sequences of all the chains in a fasta file"""
    with open(fasta_file, 'r') as f:
        return ''.join(
            [seq.rstrip() for seq in f.readlines() if seq[0] != '>'])


def get_fasta_chain_seq(fasta_file, chain_id):
    for chain in SeqIO.parse(fasta_file, 'fasta'):
        if ":{}".format(chain_id) in chain.id:
            return str(chain.seq)


def get_heavy_seq_len(fasta_file):
    h_len = len(get_fasta_chain_seq(fasta_file, "H"))

    return h_len


def lev_distance(s1, s2):
    oh1 = one_hot_seq(s1)
    oh2 = one_hot_seq(s2)

    return torch.sum((oh1 - oh2) > 0).item()
