from typing import Union
from tqdm import tqdm
import numpy as np
import torch

from deepab.models.AbResNet import AbResNet
from deepab.models.ModelEnsemble import ModelEnsemble
from deepab.util.model_out import get_inputs_from_fasta, bin_matrix
from deepab.util.util import load_full_seq, one_hot_seq, lev_distance, _aa_dict, _rev_aa_dict


def generate_pssm(
    model: AbResNet,
    fasta_file: str,
    teacher_forcing_ratio: float = 1.,
):
    wt_seq = get_inputs_from_fasta(fasta_file)
    pssm = model.get_lstm_pssm(wt_seq,
                               teacher_forcing_ratio=teacher_forcing_ratio)
    pssm = pssm[0].transpose(0, 1).numpy()

    return pssm


def get_cce_for_inputs(
    model: Union[ModelEnsemble, AbResNet],
    inputs: torch.Tensor,
):

    with torch.no_grad():
        cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-999)
        mut_logits = model(inputs)

        mut_geometries = torch.stack(
            [bin_matrix(logits[0]) for logits in mut_logits])
        wt_local_mask = mut_geometries[0] < 10

        mut_losses = [
            cross_entropy(logits[:, :, wt_local_mask],
                          label[:, wt_local_mask]).detach()
            for logits, label in zip(mut_logits, mut_geometries.unsqueeze(1))
        ]
        cce = sum(mut_losses).item()

    return cce


def get_fasta_cce(
    model: Union[ModelEnsemble, AbResNet],
    fasta_file: str,
    device: str = None,
):
    inputs = get_inputs_from_fasta(fasta_file)
    if type(device) != type(None):
        model = model.to(device)
        inputs = inputs.to(device)

    cce = get_cce_for_inputs(model, inputs)

    return cce


def get_dcce(
    model: Union[ModelEnsemble, AbResNet],
    des_fasta: str,
    wt_fasta: str,
    device: str,
):
    des_cce = get_fasta_cce(model, des_fasta, device)
    wt_cce = get_fasta_cce(model, wt_fasta, device)

    pssm = generate_pssm(model.models[0], wt_fasta, teacher_forcing_ratio=1)
    nl_pssm = -np.log(pssm)

    dcce = des_cce - wt_cce

    return dcce, des_cce, wt_cce


def get_ld_balanced_mutants(
    wt_fasta: str,
    mut_positions,
    num_seqs: int = 500,
    min_ld: int = 1,
    max_ld: int = None,
):
    if max_ld == None:
        max_ld = len(mut_positions)

    wt_seq = load_full_seq(wt_fasta)
    mut_seqs = []
    for _ in range(num_seqs):
        ld_i = int(np.random.uniform(low=min_ld - 1, high=max_ld)) + 1
        edit_pos = np.random.choice(mut_positions, size=ld_i, replace=False)
        edit_aa = torch.randint(0, 20, (len(edit_pos), ))

        mut_seq = wt_seq
        for pos in edit_pos:
            wt_aa = int(_aa_dict[wt_seq[pos]])
            edit_aa = np.random.choice(list(set(range(20)) - set([wt_aa])),
                                       size=1,
                                       replace=False)[0]
            mut_seq = mut_seq[:pos] + _rev_aa_dict[str(
                edit_aa)] + mut_seq[pos + 1:]

        mut_seqs.append(mut_seq)

    return mut_seqs


def get_ld_balanced_cce(
    model,
    wt_fasta,
    mut_positions,
    device,
    num_seqs=500,
    min_ld=1,
    max_ld=None,
):
    wt_inputs = get_inputs_from_fasta(wt_fasta)
    wt_seq = load_full_seq(wt_fasta)
    mut_seqs = get_ld_balanced_mutants(wt_fasta,
                                       mut_positions,
                                       num_seqs=num_seqs,
                                       min_ld=min_ld,
                                       max_ld=max_ld)

    ld_vals = []
    cce_vals = []
    for mut_seq in tqdm(mut_seqs):
        mut_inputs = wt_inputs.clone()
        mut_inputs[:, :20] = one_hot_seq(mut_seq).transpose(0, 1)

        cce = get_cce_for_inputs(model, mut_inputs.to(device))

        ld_vals.append(lev_distance(wt_seq, mut_seq))
        cce_vals.append(cce)

    ld_cce_mat = torch.stack(
        [torch.tensor(ld_vals).float(),
         torch.tensor(cce_vals)])

    return mut_seqs, ld_cce_mat