import math


def get_dist_bins(num_bins, interval=0.5):
    bins = [(interval * i, interval * (i + 1)) for i in range(num_bins - 1)]
    bins.append((bins[-1][1], float('Inf')))
    return bins


def get_dihedral_bins(num_bins, rad=False):
    first_bin = -180
    bin_width = 2 * 180 / num_bins
    bins = [(first_bin + bin_width * i, first_bin + bin_width * (i + 1))
            for i in range(num_bins)]

    if rad:
        bins = deg_bins_to_rad(bins)

    return bins


def get_planar_bins(num_bins, rad=False):
    first_bin = 0
    bin_width = 180 / num_bins
    bins = [(first_bin + bin_width * i, first_bin + bin_width * (i + 1))
            for i in range(num_bins)]

    if rad:
        bins = deg_bins_to_rad(bins)

    return bins


def deg_bins_to_rad(bins):
    return [(v[0] * math.pi / 180, v[1] * math.pi / 180) for v in bins]


def get_bin_values(bins):
    bin_values = [t[0] for t in bins]
    bin_width = (bin_values[2] - bin_values[1]) / 2
    bin_values = [v + bin_width for v in bin_values]
    bin_values[0] = bin_values[1] - 2 * bin_width
    return bin_values
