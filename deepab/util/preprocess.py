def bin_value_matrix(value_mat, bins):
    binned_matrix = value_mat.clone().detach().long()
    for i, (lower_bound, upper_bound) in enumerate(bins):
        bin_mask = (value_mat >= lower_bound).__and__(value_mat <= upper_bound)
        binned_matrix[bin_mask] = i

    return binned_matrix
