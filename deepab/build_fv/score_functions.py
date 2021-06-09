import pyrosetta


def get_sf_cen(
    constraint_scale: float = 1
) -> pyrosetta.rosetta.core.scoring.ScoreFunction:
    """
    Get score function for centroid minimization stage
    """

    # Score weights are adopted from trRosetta minimization protocol and found to work well
    sf = pyrosetta.rosetta.core.scoring.ScoreFunction()
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cen_hb, 5.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.rama, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.omega, 0.5)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.vdw, 1.0)
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint,
        5.0 * constraint_scale)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint,
                  4.0 * constraint_scale)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.angle_constraint,
                  4.0 * constraint_scale)

    return sf


def get_sf_cart(
    constraint_scale: float = 1
) -> pyrosetta.rosetta.core.scoring.ScoreFunction:
    """
    Get score function for full-atom cartesian minimization stage
    """

    # Score weights are adopted from trRosetta minimization protocol and found to work well
    sf = pyrosetta.rosetta.core.scoring.ScoreFunction()
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb, 3.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb, 3.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.rama, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.omega, 0.5)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.vdw, 0.5)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.1)
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint,
        5.0 * constraint_scale)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint,
                  4.0 * constraint_scale)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.angle_constraint,
                  4.0 * constraint_scale)

    return sf


def get_sf_fa(
    constraint_scale: float = 1
) -> pyrosetta.rosetta.core.scoring.ScoreFunction:
    """
    Get score function for full-atom minimization and scoring
    """

    sf = pyrosetta.create_score_function('ref2015')
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint,
        5.0 * constraint_scale)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint,
                  1.0 * constraint_scale)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.angle_constraint,
                  1.0 * constraint_scale)

    return sf