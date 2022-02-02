import os
import argparse
import tempfile
from glob import glob
from tqdm import tqdm
import torch
import numpy as np
from Bio import SeqIO

import deepab
from deepab.models.AbResNet import load_model
from deepab.models.ModelEnsemble import ModelEnsemble
from deepab.analysis.design_metrics import *
from deepab.util.pdb import cdr_indices, pdb2fasta, renumber_pdb, write_pdb_bfactor

cdr_names = ["h1", "h2", "h3", "l1", "l2", "l3"]
branch_names = ["ca", "cb", "no", "omega", "theta", "phi"]


def get_sequence_pairs(h_fasta_file, l_fasta_file):
    sequence_pairs = {}

    with open(h_fasta_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            sequence_pairs[record.id] = {"H": str(record.seq)}

    with open(l_fasta_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            sequence_pairs[record.id]["L"] = str(record.seq)

    has_mismatch_seq = False
    for id, seq_dict in sequence_pairs.items():
        if not "H" in seq_dict:
            print("Found heavy seq but not light seq for ID {}".format(id))
            has_mismatch_seq = True
        if not "L" in seq_dict:
            print("Found light seq but not heavy seq for ID {}".format(id))
            has_mismatch_seq = True

    if has_mismatch_seq:
        exit("Found mismatched sequences. Exiting.")

    return sequence_pairs


def score_designs(model, wt_fasta, h_fasta, l_fasta, device):
    sequence_pairs = get_sequence_pairs(h_fasta, l_fasta)

    wt_cce = get_fasta_cce(model, wt_fasta, device)

    for id, seq_pair in tqdm(sequence_pairs.items(),
                             total=len(sequence_pairs)):
        h_seq, l_seq = seq_pair["H"], seq_pair["L"]

        temp_fasta = tempfile.NamedTemporaryFile().name
        with open(temp_fasta, "w") as f:
            f.write(">:H\n{}\n>:L\n{}\n".format(h_seq, l_seq))

        des_cce = get_fasta_cce(model, temp_fasta, device)
        dcce = des_cce - wt_cce

        seq_pair["dCCE"] = dcce

    return sequence_pairs


def _get_args():
    """Gets command line arguments"""
    project_path = os.path.abspath(os.path.join(deepab.__file__, "../.."))

    desc = ("""
        Script for calculating design metrics for antibody Fv sequences.
        """)
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "wt_fasta_file",
        type=str,
        help=
        "Fasta file containing wild type Fv heavy and light chain sequences.")
    parser.add_argument("heavy_fasta_file",
                        type=str,
                        help="Fasta file containing Fv heavy chain sequences.")
    parser.add_argument("light_fasta_file",
                        type=str,
                        help="Fasta file containing Fv light chain sequences.")

    parser.add_argument("out_file",
                        type=str,
                        default=None,
                        help="File to save calculated design metrics.")

    default_model_dir = "trained_models/ensemble_abresnet"
    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_model_dir,
        help="Directory containing pretrained model files (in .pt format).")

    parser.add_argument('--use_gpu', default=False, action="store_true")

    return parser.parse_args()


def _cli():
    args = _get_args()

    wt_fasta_file = args.wt_fasta_file
    h_fasta_file = args.heavy_fasta_file
    l_fasta_file = args.light_fasta_file
    out_file = args.out_file
    model_dir = args.model_dir

    device_type = 'cuda' if torch.cuda.is_available(
    ) and args.use_gpu else 'cpu'
    device = torch.device(device_type)

    model_files = list(glob(os.path.join(model_dir, "*.pt")))
    if len(model_files) == 0:
        exit("No model files found at: {}".format(model_dir))

    model = ModelEnsemble(model_files=model_files,
                          load_model=load_model,
                          eval_mode=True,
                          device=device)

    sequence_pairs = score_designs(model,
                                   wt_fasta_file,
                                   h_fasta_file,
                                   l_fasta_file,
                                   device=device)

    with open(out_file, "w") as f:
        for id, seq_pair in sequence_pairs.items():
            f.write("{},{},{},{}\n".format(id, seq_pair["H"], seq_pair["L"],
                                           seq_pair["dCCE"]))


if __name__ == '__main__':
    _cli()