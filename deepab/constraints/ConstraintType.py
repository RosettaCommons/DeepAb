import enum


class ConstraintType(enum.Enum):
    ca_distance = 1
    cb_distance = 2
    no_distance = 3
    omega_dihedral = 4
    theta_dihedral = 5
    phi_planar = 6
