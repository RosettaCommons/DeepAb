import pyrosetta


def get_ab_metrics(pose_1, pose_2):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
    pose_i2 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_2)

    results = pyrosetta.rosetta.protocols.antibody.cdr_backbone_rmsds(
        pose_1, pose_2, pose_i1, pose_i2)

    results_labels = [
        'ocd', 'frh_rms', 'h1_rms', 'h2_rms', 'h3_rms', 'frl_rms', 'l1_rms',
        'l2_rms', 'l3_rms'
    ]
    results_dict = {}
    for i in range(9):
        results_dict[results_labels[i]] = results[i + 1]

    return results_dict
