import os
import torch
import pyrosetta

from .mds import build_fv_mds
from .score_functions import *
from .utils import get_constraint_set_mover, resolve_clashes, disulfidize
from deepab.constraints import ConstraintType, get_constraint_residue_pairs, get_filtered_constraint_defs
from deepab.constraints.custom_filters import hb_dist_filter
from deepab.constraints.rosetta_constraint_generators import logit_to_energy
from deepab.models.AbResNet import AbResNet
from deepab.util.model_out import get_probs_from_model, bin_matrix, binned_mat_to_values
from deepab.util.get_bins import get_dist_bins, get_dihedral_bins, get_planar_bins
from deepab.util.util import get_heavy_seq_len
from typing import Dict, List


def build_initial_fv(fasta_file: str,
                     mds_pdb_file: str,
                     model: AbResNet,
                     mask_distant_orientations: bool = True,
                     single_chain: bool = False,
                     device: str = None) -> None:
    """
    Build initial Fv structure from AbResNet outputs via MDS
    """

    masked_bin_num = 1 if mask_distant_orientations else 0
    bins = [
        get_dist_bins(model._num_out_bins),
        get_dist_bins(model._num_out_bins),
        get_dist_bins(model._num_out_bins),
        get_dihedral_bins(model._num_out_bins - masked_bin_num, rad=True),
        get_dihedral_bins(model._num_out_bins - masked_bin_num, rad=True),
        get_planar_bins(model._num_out_bins - masked_bin_num, rad=True)
    ]

    probs = get_probs_from_model(model, fasta_file, device=device)
    pred_bin_mats = [bin_matrix(p, are_logits=False) for p in probs]
    pred_value_mats = [
        binned_mat_to_values(pred_bin_mats[i], bins[i])
        for i in range(len(pred_bin_mats))
    ]

    dist_mask = pred_value_mats[1] < bins[1][-1][0]

    build_fv_mds(fasta_file,
                 mds_pdb_file,
                 pred_value_mats[1],
                 pred_value_mats[3],
                 pred_value_mats[4],
                 pred_value_mats[5],
                 mask=dist_mask,
                 single_chain=single_chain)

    pose = pyrosetta.pose_from_pdb(mds_pdb_file)
    disulfidize(pose, pred_value_mats[1])
    pose.dump_pdb(mds_pdb_file)


def get_cst_defs(model: torch.nn.Module,
                 fasta_file: str,
                 device: str = None) -> str:
    """
    Generate standard constraint files for Fv builder
    """

    residue_pairs = get_constraint_residue_pairs(model,
                                                 fasta_file,
                                                 use_logits=True,
                                                 device=device)

    all_cst_defs = get_filtered_constraint_defs(
        residue_pairs=residue_pairs,
        threshold=0.1,
        constraint_types=[
            ConstraintType.cb_distance, ConstraintType.ca_distance,
            ConstraintType.omega_dihedral, ConstraintType.theta_dihedral,
            ConstraintType.phi_planar
        ],
        prob_to_energy=logit_to_energy)
    hb_cst_defs = get_filtered_constraint_defs(
        residue_pairs=residue_pairs,
        local=True,
        threshold=0.3,
        constraint_types=[ConstraintType.no_distance],
        constraint_filters=[hb_dist_filter],
        prob_to_energy=logit_to_energy)

    # os.system("cat {} >> {}".format(all_cst_file, hb_cst_file))
    hb_cst_defs.extend(all_cst_defs)

    # one combined in-memory constraint set
    return hb_cst_defs


def get_centroid_min_mover(
        max_iter: int = 1000,
        repeats: int = 6) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create centroid minimization mover
    """

    sf = get_sf_cen()
    sf_cart = get_sf_cart()

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)

    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(max_iter)

    min_mover_cart = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf_cart, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover_cart.max_iter(max_iter)
    min_mover_cart.cartesian(True)

    repeat_mover = pyrosetta.rosetta.protocols.moves.RepeatMover(
        min_mover, repeats)

    centroid_fold_mover = pyrosetta.rosetta.protocols.moves.SequenceMover()
    centroid_fold_mover.add_mover(repeat_mover)
    centroid_fold_mover.add_mover(min_mover_cart)

    return centroid_fold_mover


def get_fa_relax_mover(
        max_iter: int = 200) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create full-atom relax mover
    """

    sf_fa = get_sf_fa()

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa)
    relax.max_iter(max_iter)
    relax.dualspace(True)
    relax.set_movemap(mmap)
    relax.ramp_down_constraints(True)

    return relax


def get_fa_min_mover(
        max_iter: int = 1000) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create full-atom minimization mover
    """

    sf_fa = get_sf_fa()
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 1)
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0)

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf_fa, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(max_iter)
    min_mover.cartesian(True)

    return min_mover


def refine_fv(in_pdb_file: str,
              out_pdb_file: str,
              cst_defs,
              verbose: bool = False) -> float:
    """
    Run constrained minimization protocol on initial pdb file and return final score
    """

    ############################################################################
    # Load initial pose
    ############################################################################
    pose = pyrosetta.pose_from_pdb(in_pdb_file)

    switch_cen = pyrosetta.rosetta.protocols.simple_moves.SwitchResidueTypeSetMover(
        "centroid")
    switch_fa = pyrosetta.rosetta.protocols.simple_moves.SwitchResidueTypeSetMover(
        "fa_standard")
    switch_cen.apply(pose)

    # AMW TODO: could pass pose here if we must
    #     ConstraintSetOP ConstraintIO::read_constraints(
    # 	std::istream & data,
    # 	ConstraintSetOP cset,
    # 	pose::Pose const& pose,
    # 	bool const force_pdb_info_mapping// = false
    # ) {
    # from io import StringIO
    csts = pyrosetta.rosetta.core.scoring.constraints.ConstraintIO.read_constraints(
        # StringIO('\n'.join(cst_files['hb_cst_file'])),
        pyrosetta.rosetta.std.stringstream('\n'.join(cst_defs)),
        pyrosetta.rosetta.core.scoring.constraints.ConstraintSet(),
        pose)
    csm = get_constraint_set_mover(csts)
    csm.apply(pose)

    ############################################################################
    # Apply global centroid movers
    ############################################################################
    if verbose:
        print('centroid minimize...')

    centroid_fold_mover = get_centroid_min_mover()
    switch_cen.apply(pose)
    centroid_fold_mover.apply(pose)

    resolve_clashes(pose)

    ############################################################################
    # Apply full atom relax
    ############################################################################
    if verbose:
        print('full atom relax...')

    fa_relax_mover = get_fa_relax_mover()
    switch_fa.apply(pose)
    fa_relax_mover.apply(pose)

    ############################################################################
    # Apply full atom refinement
    ############################################################################
    if verbose:
        print("full atom minimize...")

    fa_min_mover = get_fa_min_mover()
    fa_min_mover.apply(pose)

    pose.dump_pdb(out_pdb_file)

    sf_fa_cst = get_sf_fa()
    score = sf_fa_cst(pose)

    return score