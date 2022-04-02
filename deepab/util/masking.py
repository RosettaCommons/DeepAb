MASK_VALUE = -999


def make_square_mask(v_mask):
    sq_mask = v_mask.expand((len(v_mask), len(v_mask)))
    sq_mask = sq_mask & sq_mask.transpose(0, 1)
    return sq_mask