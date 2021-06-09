import torch
import torch.nn as nn


class OuterConcatenation2D(nn.Module):
    """Transforms sequential data to pairwise data using an outer concatenation (similar to an outer product)."""
    def __init__(self):
        super(OuterConcatenation2D, self).__init__()

    def forward(self, x):
        """
        :param x: A FloatTensor to propagate forward
        :type x: torch.Tensor
        """
        if len(x.shape) != 3:
            raise ValueError('Expected three dimensional shape, got shape {}'.format(x.shape))

        # Switch shape from [batch, filter/channel, timestep/length] to [batch, timestep/length, filter/channel]
        x = torch.transpose(x, 1, 2)

        # Vertical expansion, convert to bxLxLxn where out_tensor[b][i][j] = in_matrix[b][i]
        vert_expansion = x.clone()
        vert_expansion.unsqueeze_(2)
        vert_expansion = vert_expansion.expand(vert_expansion.shape[0], vert_expansion.shape[1],
                                               vert_expansion.shape[1], vert_expansion.shape[3])

        # For every b, i, j pair, append in_matrix[b][j] to out_tensor[b][i][j]
        x_shape = x.shape
        pair = x
        pair.unsqueeze_(1)
        pair = pair.expand(pair.shape[0], x_shape[1], pair.shape[2], pair.shape[3])
        out_tensor = torch.cat([vert_expansion, pair], dim=3)

        # Switch shape from [batch, timestep/length_i, timestep/length_j, filter/channel]
        #                to [batch, filter/channel, timestep/length_i, timestep/length_j]
        out_tensor = torch.einsum('bijc -> bcij', out_tensor)

        return out_tensor

