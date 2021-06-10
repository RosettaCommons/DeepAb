import os
import argparse
import tempfile
from datetime import datetime
from glob import glob
from tqdm.contrib.concurrent import process_map
import torch
import numpy as np
import pyrosetta

import deepab
from deepab.models.AbResNet import load_model
from deepab.analysis.attention_analysis import *
from deepab.util.pdb import cdr_indices, pdb2fasta, renumber_pdb, write_pdb_bfactor


def prog_print(text):
    print("*" * 50)
    print(text)
    print("*" * 50)


def annotate_structure(model, pdb_file, out_file):
    print()


def _get_args():
    """Gets command line arguments"""
    project_path = os.path.abspath(os.path.join(deepab.__file__, "../.."))

    desc = ("""
        Script for annotating Fv structures with attention for a given CDR loop.
        Inter-residue attention will be added to the PDB file as b-factor.
        """)
    parser = argparse.ArgumentParser(description=desc)

    # parser.add_argument("pdb_file",
    #                     type=str,
    #                     help="
    #     PDB file containing structure to be annotated.
    #     Heavy and light chain sequences should be truncated at Chothia positions 112 and 109.
    # ")

    parser.add_argument("--out_file",
                        type=str,
                        default=None,
                        help="PDB file for annotated structure.")

    default_model_file = "trained_models/ensemble_abresnet/rs0.pt"
    parser.add_argument(
        "--model_file",
        type=str,
        default=default_model_file,
        help="Pretrained model file (in .pt format) to use attention from.")

    parser.add_argument(
        "--renumber",
        default=False,
        action="store_true",
        help=
        "Convert structure to Chothia format using AbNum before annotation.")
    parser.add_argument("--cdr_loop",
                        type=str,
                        default="h3",
                        help="CDR loop to aggregate attention over.")
    parser.add_argument("--attention_branch",
                        type=str,
                        default="CA",
                        help="Output branch to use attention from.")

    return parser.parse_args()


def _cli():
    args = _get_args()

    # pdb_file = args.pdb_file
    pdb_file = "data/sample_files/4h0h.truncated.pdb"
    out_file = args.out_file
    model_file = args.model_file
    renumber = args.renumber
    cdr_loop = args.cdr_loop.lower()
    attention_branch = args.attention_branch.lower()

    device_type = 'cuda' if torch.cuda.is_available(
    ) and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    model = load_model(model_file, eval_mode=True, device=device)

    cdr_names = ["h1", "h2", "h3", "l1", "l2", "l3"]
    if not cdr_loop in cdr_names:
        exit("Provided CDR loop not recognized: {}\nMust be one of {}".format(
            cdr_loop, cdr_names))

    branch_names = ["ca", "cn", "no", "omega", "theta", "phi"]
    if not attention_branch in branch_names:
        exit("Provided attention branch not recognized: {}\nMust be one of {}".
             format(attention_branch, branch_names))

    if out_file == None:
        out_file = pdb_file

    if renumber:
        renumber_pdb(pdb_file, out_file)

    temp_fasta = tempfile.NamedTemporaryFile().name
    with open(temp_fasta, "w") as f:
        fasta_content = pdb2fasta(pdb_file)
        f.write(fasta_content)

    hw_attn_mats = get_HW_attn_for_fasta(model, temp_fasta)
    branch_attn = hw_attn_mats[branch_names.index(attention_branch)]

    cdr_i = cdr_indices(pdb_file, cdr_loop)
    cdr_attn = get_mean_range_attn(branch_attn, cdr_i)[0]

    write_pdb_bfactor(in_pdb_file=out_file,
                      out_pdb_file=out_file,
                      bfactor=cdr_attn * 100)


if __name__ == '__main__':
    _cli()