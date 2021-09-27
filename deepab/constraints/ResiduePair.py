import math
from typing import List

from .Constraint import Constraint
from .Residue import Residue


class ResiduePair():
    """
    Class containing the set of geometric distributions between two residues
    """
    def __init__(self, residue_1: Residue, residue_2: Residue,
                 constraints: List[Constraint]):
        super().__init__()

        self.residue_1 = residue_1
        self.residue_2 = residue_2
        self.constraints = constraints
        self.constraint_types = set([c.constraint_type for c in constraints])

    def get_constraints(self,
                        modal_x_min=-math.inf,
                        modal_x_max=math.inf,
                        modal_y_min=0,
                        modal_y_max=1,
                        average_x_min=-math.inf,
                        average_x_max=math.inf,
                        average_y_min=0,
                        average_y_max=1,
                        custom_filters=[]):
        filtered_constraints = []
        for c in self.constraints:
            if not (modal_x_min <= c.modal_x <= modal_x_max):
                continue
            if not (modal_y_min <= c.modal_y <= modal_y_max):
                continue
            if not (average_x_min <= c.average_x <= average_x_max):
                continue
            if not (average_y_min <= c.average_y <= average_y_max):
                continue

            pass_filters = True
            for custom_filter in custom_filters:
                if not custom_filter(self, c):
                    pass_filters = False
                    break

            if not pass_filters:
                continue

            filtered_constraints.append(c)

        return filtered_constraints