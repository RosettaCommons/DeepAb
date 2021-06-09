import torch

MASK_VALUE = -999


def get_range_all_mask(unmasked_range, seq_len, mask_1d=False):
    mask = torch.zeros(seq_len).long()
    mask[unmasked_range[0]:unmasked_range[1] + 1] = 1

    if mask_1d:
        return mask

    mask = mask.expand((seq_len, seq_len))
    mask = mask | mask.transpose(0, 1)

    return mask


def get_range_range_mask(unmasked_range, seq_len, mask_1d=False):
    mask = torch.zeros(seq_len).long()
    mask[unmasked_range[0]:unmasked_range[1] + 1] = 1

    if mask_1d:
        return mask

    mask = mask.expand((seq_len, seq_len))
    mask = mask & mask.transpose(0, 1)

    return mask


def get_ranges_all_mask(unmasked_ranges, seq_len, mask_1d=False):
    mask = torch.zeros((seq_len, seq_len)).long()
    if mask_1d:
        mask = torch.zeros(seq_len).long()

    for unmasked_range in unmasked_ranges:
        mask = mask | get_range_all_mask(
            unmasked_range, seq_len, mask_1d=mask_1d)

    return mask


def get_ranges_ranges_mask(unmasked_ranges, seq_len, mask_1d=False):
    mask = torch.zeros((seq_len, seq_len)).long()
    if mask_1d:
        mask = torch.zeros(seq_len).long()

    for unmasked_range in unmasked_ranges:
        mask = mask | get_range_range_mask(
            unmasked_range, seq_len, mask_1d=mask_1d)

    return mask


def mask_diagonal(mask):
    assert len(mask.shape) == 2

    mask[torch.eye(len(mask)) == 1] = 0

    return mask


def get_missing_value_mask(value_mat, missing_value=MASK_VALUE):
    mask = torch.ones(value_mat.shape).long()
    mask[value_mat == missing_value] = 0

    return mask


def get_prob_mask(prob_mat, prob_cutoff=0.1, mask_1d=False):
    mask = torch.ones(prob_mat.shape).long()
    mask[prob_mat < prob_cutoff] = 0

    return mask


def get_max_dist_mask(dist_value_mat, distance_cutoff=12):
    assert len(dist_value_mat.shape) == 2

    mask = torch.ones(dist_value_mat.shape).long()
    mask[dist_value_mat > distance_cutoff] = 0

    return mask


def get_min_dist_mask(dist_value_mat, distance_cutoff=4):
    assert len(dist_value_mat.shape) == 2

    mask = torch.ones(dist_value_mat.shape).long()
    mask[dist_value_mat < distance_cutoff] = 0
    mask[dist_value_mat < 0] = 1

    return mask


def get_extreme_bin_mask(binned_mat, num_bins):
    assert len(binned_mat.shape) == 2

    mask = torch.ones(binned_mat.shape).long()
    mask[binned_mat == 0] = 0
    mask[binned_mat == num_bins - 1] = 0

    return mask


def get_gly_mask(seq):
    seq_len = len(seq)
    mask = torch.ones(seq_len).long()

    for i, res in enumerate(seq):
        if res == "G":
            mask[i] == 0

    mask = mask.expand((seq_len, seq_len))
    mask = mask & mask.transpose(0, 1)

    return mask


def make_square_mask(v_mask):
    sq_mask = v_mask.expand((len(v_mask), len(v_mask)))
    sq_mask = sq_mask & sq_mask.transpose(0, 1)
    return sq_mask