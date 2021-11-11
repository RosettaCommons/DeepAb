import os
import math
from typing import Iterable, List
from tqdm import tqdm
import torch

from deepab.constraints import Constraint, ConstraintType, Residue, ResiduePair, constraint_type_generator_dict
from deepab.constraints.custom_filters import no_max_distance_filter, local_interaction_filter
from deepab.constraints.rosetta_constraint_generators import logit_to_energy
from deepab.models.AbResNet import AbResNet
from deepab.models.ModelEnsemble import ModelEnsemble
from deepab.util.get_bins import get_dist_bins, get_dihedral_bins, get_planar_bins, get_bin_values
from deepab.util.model_out import get_logits_from_model, get_probs_from_model, bin_matrix, binned_mat_to_values
from deepab.util.util import load_full_seq

model_out_constraint_dict = {
    AbResNet: [
        ConstraintType.ca_distance, ConstraintType.cb_distance,
        ConstraintType.no_distance, ConstraintType.omega_dihedral,
        ConstraintType.theta_dihedral, ConstraintType.phi_planar
    ]
}

pairwise_constraint_types = [
    ConstraintType.ca_distance, ConstraintType.cb_distance,
    ConstraintType.no_distance, ConstraintType.omega_dihedral,
    ConstraintType.theta_dihedral, ConstraintType.phi_planar
]

asymmetric_constraint_types = [
    ConstraintType.no_distance, ConstraintType.theta_dihedral,
    ConstraintType.phi_planar
]


def get_constraint_bin_value_dict(num_out_bins: int,
                                  mask_distant_orientations: bool = True):
    masked_bin_num = 1 if mask_distant_orientations else 0
    dist_bin_values = get_bin_values(get_dist_bins(num_out_bins))
    dihedral_bin_values = get_bin_values(
        get_dihedral_bins(num_out_bins - masked_bin_num, rad=True))
    planar_bin_values = get_bin_values(
        get_planar_bins(num_out_bins - masked_bin_num, rad=True))

    constraint_bin_value_dict = {
        ConstraintType.ca_distance: dist_bin_values,
        ConstraintType.cb_distance: dist_bin_values,
        ConstraintType.no_distance: dist_bin_values,
        ConstraintType.omega_dihedral: dihedral_bin_values,
        ConstraintType.theta_dihedral: dihedral_bin_values,
        ConstraintType.phi_planar: planar_bin_values
    }

    return constraint_bin_value_dict


def get_constraint_residue_pairs(model: torch.nn.Module,
                                 fasta_file: str,
                                 constraint_bin_value_dict: dict = None,
                                 mask_distant_orientations: bool = True,
                                 use_logits: bool = True,
                                 device: str = None):
    seq = load_full_seq(fasta_file)

    model_type = type(
        model) if not type(model) == ModelEnsemble else model.model_type()
    model_out_constraint_types = model_out_constraint_dict[model_type]
    if constraint_bin_value_dict == None:
        constraint_bin_value_dict = get_constraint_bin_value_dict(
            model._num_out_bins,
            mask_distant_orientations=mask_distant_orientations)

    if use_logits:
        logits = get_logits_from_model(model, fasta_file, device=device)
        preds = [l.permute(1, 2, 0).cpu() for l in logits]

        ca_dist_mat = binned_mat_to_values(
            bin_matrix(preds[0].permute(2, 0, 1), are_logits=use_logits),
            bins=get_dist_bins(model._num_out_bins))
        y_scale = 1 / (ca_dist_mat * ca_dist_mat)
    else:
        preds = get_probs_from_model(model, fasta_file)
        y_scale = torch.ones((len(seq), len(seq)))

    residue_pairs = []
    for i in tqdm(range(len(seq))):
        residue_i = Residue(identity=seq[i], index=i + 1)
        for j in range(i):
            residue_j = Residue(identity=seq[j], index=j + 1)

            ij_constraints = []
            for pred_i, constraint_type in enumerate(
                    model_out_constraint_types):

                if constraint_type in pairwise_constraint_types:
                    if preds[pred_i][i, j].argmax().item() >= len(
                            constraint_bin_value_dict[constraint_type]):
                        continue
                    ij_constraints += [
                        Constraint(
                            constraint_type=constraint_type,
                            residue_1=residue_i,
                            residue_2=residue_j,
                            x_vals=constraint_bin_value_dict[constraint_type],
                            y_vals=preds[pred_i][i, j]
                            [:len(constraint_bin_value_dict[constraint_type])],
                            are_logits=use_logits,
                            y_scale=y_scale[i, j])
                    ]

                    if constraint_type in asymmetric_constraint_types:
                        ij_constraints += [
                            Constraint(
                                constraint_type=constraint_type,
                                residue_1=residue_j,
                                residue_2=residue_i,
                                x_vals=constraint_bin_value_dict[
                                    constraint_type],
                                y_vals=preds[pred_i][j, i][:len(
                                    constraint_bin_value_dict[constraint_type]
                                )],
                                are_logits=use_logits,
                                y_scale=y_scale[i, j])
                        ]

            residue_pairs.append(
                ResiduePair(residue_1=residue_i,
                            residue_2=residue_j,
                            constraints=ij_constraints))

    return residue_pairs


def get_filtered_constraint_defs(residue_pairs: List[ResiduePair],
                                 threshold: float = 0.1,
                                 res_range: Iterable = None,
                                 max_separation: int = math.inf,
                                 local: bool = False,
                                 heavy_seq_len=None,
                                 heavy_only: bool = False,
                                 light_only: bool = False,
                                 interchain: bool = False,
                                 constraint_types: List[ConstraintType] = None,
                                 constraint_filters: List = None,
                                 prob_to_energy=logit_to_energy):
    """
    returns an in-memory list of text constraint definitions
    """

    # Use default constraint filters if none are provided
    if constraint_filters is None:
        constraint_filters = [
            # Filter out angles on pairs with a glycine
            lambda _, c:
            (c.constraint_type == ConstraintType.cb_distance or
             (c.residue_1.identity != "G" and c.residue_2.identity != "G")),
            # Filter out pairs predicted to be in last distance bin
            no_max_distance_filter,
            # Filter out constraints with greater sequence separation than max
            lambda rp, _: abs(rp.residue_1.index - rp.residue_2.index
                              ) <= max_separation,
            # Filter out CB distances on pairs with a glycine
            lambda _, c: not (
                c.constraint_type == ConstraintType.cb_distance and
                (c.residue_1.identity == "G" or c.residue_2.identity == "G"))
        ]
    if not res_range == None:
        assert len(res_range) == 2
        constraint_filters.append(
            # Filter out pairs without residue in given range
            lambda rp, _: (res_range[0] <= rp.residue_1.index - 1 <= res_range[
                1] or res_range[0] <= rp.residue_2.index - 1 <= res_range[1]))
    if local:
        constraint_filters.append(local_interaction_filter)
    if heavy_only and not heavy_seq_len == None:
        constraint_filters.append(
            # Filter out light chain pairs
            lambda rp, _: (rp.residue_1.index - 1 < heavy_seq_len and rp.
                           residue_2.index - 1 < heavy_seq_len))
    if light_only and not heavy_seq_len == None:
        constraint_filters.append(
            # Filter out light chain pairs
            lambda rp, _: (rp.residue_2.index - 1 >= heavy_seq_len and rp.
                           residue_1.index - 1 >= heavy_seq_len))
    if interchain and not heavy_seq_len == None:
        constraint_filters.append(
            # Filter out intra-chain pairs
            lambda rp, _: (rp.residue_1.index - 1 <= heavy_seq_len and rp.
                           residue_2.index - 1 > heavy_seq_len) or
            (rp.residue_2.index - 1 <= heavy_seq_len and rp.residue_1.index - 1
             > heavy_seq_len))
    if not constraint_types == None:
        constraint_filters.append(
            lambda _, c: c.constraint_type in constraint_types)

    constraints = []
    for residue_pair in residue_pairs:
        constraints += residue_pair.get_constraints(
            custom_filters=constraint_filters)

    constraints = [c for c in constraints if c.modal_y >= threshold]

    return [
        constraint_type_generator_dict[c.constraint_type](
            c, prob_to_energy=prob_to_energy) for c in constraints
    ]
