from .Constraint import Constraint
from .ConstraintType import ConstraintType
from .ResiduePair import ResiduePair


def no_max_distance_filter(residue_pair: ResiduePair, _: Constraint):
    """
    Filter on ResiduePair that returns false if the predicted distance
    falls in the last distance bin
    """

    if ConstraintType.cb_distance in residue_pair.constraint_types:
        cb_constraint = [
            c for c in residue_pair.constraints
            if c.constraint_type == ConstraintType.cb_distance
        ][0]

        if cb_constraint.modal_x == cb_constraint.x_vals[-1]:
            return False

    return True


def local_interaction_filter(residue_pair: ResiduePair,
                             _: Constraint,
                             local_distance: float = 12):
    """
    Filter on ResiduePair that returns false if the predicted distance
    is greater than 12 Å (default)
    """

    if ConstraintType.ca_distance in residue_pair.constraint_types:
        ca_constraint = [
            c for c in residue_pair.constraints
            if c.constraint_type == ConstraintType.ca_distance
        ][0]

        if ca_constraint.modal_x > local_distance:
            return False
    elif ConstraintType.cb_distance in residue_pair.constraint_types:
        cb_constraint = [
            c for c in residue_pair.constraints
            if c.constraint_type == ConstraintType.cb_distance
        ][0]

        if cb_constraint.modal_x > local_distance:
            return False

    return True


def hb_dist_filter(_: ResiduePair, constraint: Constraint):
    """
    Filter on constraint that returns false for no_distance constraints
    with distance greater than 5 Å
    Note: 5 Å selected to provide generous cutoff for hbonds
    """

    hbond_distance = 5
    if constraint.constraint_type == ConstraintType.no_distance and constraint.modal_x < hbond_distance:
        return True

    return False