import math
from typing import Union
import torch

from .ConstraintType import ConstraintType
from .Residue import Residue


class Constraint():
    """
    Class containing predicted geometric distribution between pair of residues
    """
    def __init__(self,
                 constraint_type: ConstraintType,
                 residue_1: Residue,
                 residue_2: Residue,
                 x_vals: Union[list, torch.Tensor],
                 y_vals: Union[list, torch.Tensor],
                 are_logits: bool = False,
                 y_scale: float = 1):

        super().__init__()

        assert len(x_vals) == len(y_vals)

        self.constraint_type = constraint_type
        self.residue_1 = residue_1
        self.residue_2 = residue_2
        self.x_vals = torch.Tensor(x_vals)
        self.y_vals = torch.Tensor(y_vals)

        if are_logits:
            y_probs = torch.nn.Softmax(dim=-1)(self.y_vals)
        else:
            y_probs = self.y_vals

        self.y_vals = y_vals * y_scale

        self.bin_width = x_vals[1] - x_vals[0]

        modal_i = torch.argmax(y_probs)
        self.modal_x = self.x_vals[modal_i].item()
        self.modal_y = y_probs[modal_i].item()

        average_i = torch.round(
            torch.sum(torch.mul(torch.arange(len(y_probs)).float(),
                                y_probs))).int()
        self.average_x = self.x_vals[average_i].item()
        self.average_y = y_probs[average_i].item()
