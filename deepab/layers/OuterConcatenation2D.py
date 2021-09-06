import torch
import torch.nn as nn
from einops import repeat


class OuterConcatenation2D(nn.Module):
    """Transforms sequential data to pairwise data using an outer concatenation (similar to an outer product)."""
    def __init__(self):
        super(OuterConcatenation2D, self).__init__()

    def forward(self, x: torch.FloatTensor):
        if len(x.shape) != 3:
            raise ValueError(
                'Expected three dimensional shape, got shape {}'.format(
                    x.shape))

        seq_len = x.shape[-1]
        row_exp = repeat(x, "b c l -> b c x l", x=seq_len)
        col_exp = repeat(x, "b c l -> b c l x", x=seq_len)
        out = torch.cat([col_exp, row_exp], dim=1)

        return out
