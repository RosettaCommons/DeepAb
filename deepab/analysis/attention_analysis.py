from typing import Dict, Tuple
import torch

from deepab.models.AbResNet import AbResNet
from deepab.util.model_out import get_inputs_from_fasta


def get_HW_attn_for_model_input(
    model: AbResNet,
    model_input: torch.LongTensor,
):
    """
    Gets row- and column-wise attention matrices for one-hot encoded input
    """
    with torch.no_grad():
        hw_attn = model.forward_attn(model_input)

    # Taking the attention mats for the second CCA operation
    hw_attn = [(attn_r2[0, 0], attn_r2[0, 1]) for _, attn_r2 in hw_attn]

    return hw_attn


def get_HW_attn_for_fasta(
    model: AbResNet,
    fasta_file: str,
):
    """
    Gets row- and column-wise attention matrices for fasta file
    """
    model_input = get_inputs_from_fasta(fasta_file)
    hw_attn = get_HW_attn_for_model_input(model, model_input)

    return hw_attn


def get_mean_range_attn(
    attn: torch.FloatTensor,
    r: Tuple[int],
):
    """
    Calculates average attention on other whole sequence for given residue range
    """
    att_H, att_W = attn
    range_att_H = att_H[r[0]:r[1], :, r[0]:r[1]].mean(0)
    range_att_W = att_W[r[0]:r[1], r[0]:r[1]].mean(0)
    range_seq_attn = range_att_H.mean(1) + range_att_W.mean(0)

    seq_len = att_H.shape[1]
    attn_mat = torch.zeros((seq_len, seq_len))
    attn_mat[:, r[0]:r[1]] = range_att_H
    attn_mat[r[0]:r[1]] = range_att_W

    return range_seq_attn, attn_mat


def get_cdr_attn_dict(
    attn: torch.FloatTensor,
    cdr_range_dict: Dict[str, Tuple[int, int]],
):
    """
    Calculates attention on whole sequence for each CDR loop
    """
    cdr_attn_dict = {
        cdr: get_mean_range_attn(attn, r)
        for cdr, r in cdr_range_dict.items()
    }

    return cdr_attn_dict
