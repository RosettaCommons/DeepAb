import torch
import pyrosetta

from deepab.constraints import get_filtered_constraint_defs


def migrate_seq_numbering(source_pose: pyrosetta.Pose,
                          target_pose: pyrosetta.Pose) -> None:
    """
    Copy pdb_info from source_pose to target_pose
    """
    target_pose.pdb_info(source_pose.pdb_info())


def get_constraint_set_mover(
    csts: pyrosetta.rosetta.core.scoring.constraints.ConstraintSet  #,
    #**kwargs
) -> pyrosetta.rosetta.protocols.constraint_movers.ConstraintSetMover:

    if csts == None:
        print(
            "We're now operating under logic where this shouldn't be possible")
        quit()
    #    constraint_file = get_filtered_constraint_defs(**kwargs)

    csm = pyrosetta.rosetta.protocols.constraint_movers.ConstraintSetMover()
    csm.add_constraints(True)
    # AMW TODO: change this to constraint_set(constraint_set)
    csm.constraint_set(csts)

    return csm


def resolve_clashes(pose: pyrosetta.Pose) -> None:
    """
    Attempt to remove clashes from pose using simplified score function with high VDW
    """
    def get_sf_cen_vdw() -> pyrosetta.rosetta.core.scoring.ScoreFunction:
        """
        Get score function with increased VDW for centroid clash removal
        """

        # Score weights are adopted from trRosetta minimization protocol and found to work well
        sf = pyrosetta.rosetta.core.scoring.ScoreFunction()
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cen_hb, 5.0)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.rama, 1.0)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.omega, 0.5)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.vdw, 3.0)
        sf.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint, 3.0)
        sf.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint, 1.0)
        sf.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.angle_constraint, 1.0)

        return sf

    def get_sf_vdw() -> pyrosetta.rosetta.core.scoring.ScoreFunction:
        """
        Get simple score function  clash removal
        """

        # Score weights are adopted from trRosetta minimization protocol and found to work well
        sf = pyrosetta.rosetta.core.scoring.ScoreFunction()
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.rama, 1.0)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.vdw, 1.0)

        return sf

    sf1 = get_sf_cen_vdw()
    sf_vdw = get_sf_vdw()
    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)
    min_mover1 = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf1, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover1.max_iter(1000)

    for _ in range(0, 5):
        if float(sf_vdw(pose)) < 10:
            break
        min_mover1.apply(pose)


def disulfidize(pose: pyrosetta.Pose, cb_dist_mat: torch.Tensor) -> None:
    seq_len = len(pose.sequence())

    cys_seq_mask = torch.tensor([res == "C" for res in pose.sequence()]).int()
    cys_seq_mask = cys_seq_mask.expand((seq_len, seq_len))
    cys_seq_mask = cys_seq_mask & cys_seq_mask.transpose(0, 1)
    cys_seq_mask[torch.eye(seq_len) == 1] = 0

    disulfide_residues = torch.where((cb_dist_mat < 5) & cys_seq_mask.bool())
    for res1, res2 in zip(*disulfide_residues):
        res1, res2 = res1.item(), res2.item()
        disulfidize_mover = pyrosetta.rosetta.protocols.denovo_design.DisulfidizeMover(
        )
        sf = pyrosetta.create_score_function('ref2015')
        disulfidize_mover.make_disulfide(pose, res1 + 1, res2 + 1, True, sf)