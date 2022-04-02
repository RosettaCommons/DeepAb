import os
import argparse
from datetime import datetime
from glob import glob
from tqdm.contrib.concurrent import process_map
import torch
import numpy as np
import pyrosetta

import deepab
from deepab.models.AbResNet import load_model
from deepab.models.ModelEnsemble import ModelEnsemble
from deepab.build_fv.build_cen_fa import build_initial_fv, get_cst_defs, refine_fv
from deepab.metrics.rosetta_ab import get_ab_metrics
from deepab.util.pdb import renumber_pdb

init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"


def prog_print(text):
    print("*" * 50)
    print(text)
    print("*" * 50)


def refine_fv_(args):
    in_pdb_file, out_pdb_file, cst_defs = args
    pyrosetta.init(init_string)
    return refine_fv(in_pdb_file, out_pdb_file, cst_defs)


def build_structure(model,
                    fasta_file,
                    cst_defs,
                    out_dir,
                    target="pred",
                    num_decoys=5,
                    num_procs=1,
                    single_chain=False,
                    device=None):
    decoy_dir = os.path.join(out_dir, "decoys")
    os.makedirs(decoy_dir, exist_ok=True)

    prog_print("Creating MDS structure")
    pyrosetta.init(init_string)
    mds_pdb_file = os.path.join(decoy_dir, "{}.mds.pdb".format(target))
    build_initial_fv(fasta_file,
                     mds_pdb_file,
                     model,
                     single_chain=single_chain,
                     device=device)

    prog_print("Creating decoys structures")
    decoy_pdb_pattern = os.path.join(decoy_dir,
                                     "{}.deepab.{{}}.pdb".format(target))
    refine_args = [(mds_pdb_file, decoy_pdb_pattern.format(i), cst_defs)
                   for i in range(num_decoys)]
    decoy_scores = process_map(refine_fv_, refine_args, max_workers=num_procs)

    best_decoy_i = np.argmin(decoy_scores)
    best_decoy_pdb = decoy_pdb_pattern.format(best_decoy_i)
    out_pdb = os.path.join(out_dir, "{}.deepab.pdb".format(target))
    os.system("cp {} {}".format(best_decoy_pdb, out_pdb))

    return out_pdb


def _get_args():
    """Gets command line arguments"""
    project_path = os.path.abspath(os.path.join(deepab.__file__, "../.."))

    desc = ('''
        Script for predicting antibody Fv structures from heavy and light chain sequences.
        ''')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("fasta_file",
                        type=str,
                        help="""
        Fasta file containing Fv heavy and light chain sequences.
        Heavy and light chain sequences should be truncated at Chothia positions 112 and 109.
    """)

    now = str(datetime.now().strftime('%y-%m-%d_%H:%M:%S'))
    default_pred_dir = os.path.join(project_path, "pred_{}".format(now))
    parser.add_argument("--pred_dir",
                        type=str,
                        default=default_pred_dir,
                        help="Directory where results should be saved.")

    default_model_dir = "trained_models/ensemble_abresnet"
    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_model_dir,
        help="Directory containing pretrained model files (in .pt format).")

    parser.add_argument("--target",
                        type=str,
                        default="pred",
                        help="Identifier for predicted structure naming.")
    parser.add_argument(
        "--decoys",
        type=int,
        default=5,
        help=
        "Number of decoys to create. The lowest energy decoy will be selected as final predicted structure."
    )
    parser.add_argument(
        "--num_procs",
        type=int,
        default=5,
        help=
        "Maximum number of parallel processes that should be used for creating decoys."
    )
    parser.add_argument(
        "--renumber",
        default=False,
        action="store_true",
        help="Convert final predicted structure to Chothia format using AbNum."
    )
    parser.add_argument("--single_chain",
                        default=False,
                        action="store_true",
                        help="Predict for fasta with only one chain")
    parser.add_argument(
        "--native_pdb",
        type=str,
        default=None,
        help="Native PDB in Chothia format for measuring RMSDs.")
    parser.add_argument("--use_gpu",
                        default=False,
                        action="store_true",
                        help="Run model prediction on GPU.")

    return parser.parse_args()


def _cli():
    args = _get_args()

    fasta_file = args.fasta_file
    pred_dir = args.pred_dir
    model_dir = args.model_dir
    target = args.target
    decoys = args.decoys
    num_procs = args.num_procs
    renumber = args.renumber
    single_chain = args.single_chain
    native_pdb = args.native_pdb

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

    prog_print("Generating constraints")
    cst_defs = get_cst_defs(model, fasta_file, device=device)

    if decoys > 0:
        pred_pdb = build_structure(model,
                                   fasta_file,
                                   cst_defs,
                                   pred_dir,
                                   target=target,
                                   num_decoys=decoys,
                                   num_procs=num_procs,
                                   single_chain=single_chain,
                                   device=device)

        if renumber:
            renumber_pdb(pred_pdb, pred_pdb)

    if native_pdb is not None and os.path.exists(native_pdb):
        pose = pyrosetta.pose_from_pdb(pred_pdb)
        native_pose = pyrosetta.pose_from_pdb(native_pdb)

        metrics = get_ab_metrics(pose, native_pose)

        print("Metrics")
        for metric_name, metric_value in metrics.items():
            print("{}{}".format(metric_name.ljust(12), round(metric_value, 3)))


if __name__ == '__main__':
    _cli()
