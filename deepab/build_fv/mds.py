import torch
import numpy as np
from sklearn.manifold import MDS

from deepab.util.masking import MASK_VALUE
from deepab.util.util import _aa_1_3_dict, get_heavy_seq_len, load_full_seq

ARBITRARILY_LARGE_VALUE = 999


def place_fourth_atom(a_coord: torch.Tensor, b_coord: torch.Tensor,
                      c_coord: torch.Tensor, length: torch.Tensor,
                      planar: torch.Tensor,
                      dihedral: torch.Tensor) -> torch.Tensor:
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a fourth coord
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).expand(bc_vec.shape).cross(bc_vec)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral)
    ]

    d_coord = c_coord + sum([m * d for m, d in zip(m_vec, d_vec)])

    return d_coord


def calc_dist_mat(a_coord: torch.Tensor,
                  b_coord: torch.Tensor) -> torch.Tensor:
    """
    Calculate a distance matrix between tensors of coords
    """
    assert a_coord.shape == b_coord.shape
    mat_shape = (len(a_coord), len(a_coord), 3)

    a_coord = a_coord.unsqueeze(0).expand(mat_shape)
    b_coord = b_coord.unsqueeze(1).expand(mat_shape)

    dist_mat = (a_coord - b_coord).norm(dim=-1)

    return dist_mat


def calc_dihedral(a_coord: torch.Tensor, b_coord: torch.Tensor,
                  c_coord: torch.Tensor,
                  d_coord: torch.Tensor) -> torch.Tensor:
    """
    Calculate a dihedral between tensors of coords
    """
    b1 = a_coord - b_coord
    b2 = b_coord - c_coord
    b3 = c_coord - d_coord

    n1 = torch.cross(b1, b2)
    n1 = torch.div(n1, n1.norm(dim=-1, keepdim=True))
    n2 = torch.cross(b2, b3)
    n2 = torch.div(n2, n2.norm(dim=-1, keepdim=True))
    m1 = torch.cross(n1, torch.div(b2, b2.norm(dim=-1, keepdim=True)))

    dihedral = torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    return dihedral


def calc_planar(a_coord: torch.Tensor, b_coord: torch.Tensor,
                c_coord: torch.Tensor) -> torch.Tensor:
    """
    Calculate a planar angle between tensors of coords
    """
    v1 = a_coord - b_coord
    v2 = c_coord - b_coord

    a = (v1 * v2).sum(-1)
    b = v1.norm(dim=-1) * v2.norm(dim=-1)

    planar = torch.acos(a / b)

    return planar


def fix_chirality(coords: torch.Tensor) -> torch.Tensor:
    """
    Check chirality of protein backbone and return mirror if incorrect
    """
    n_coords, ca_coords, c_coords = coords.reshape(-1, 3, 3).permute(1, 0, 2)
    phi_mu = calc_dihedral(c_coords[:-1], n_coords[1:], ca_coords[1:],
                           c_coords[1:]).mean()

    if phi_mu > 0:
        return coords * torch.tensor([1, 1, -1]).double()
    else:
        return coords


def fix_bond_lengths(
        dist_mat: torch.Tensor,
        bond_lengths: torch.Tensor,
        delim: int = None,
        delim_value: float = ARBITRARILY_LARGE_VALUE) -> torch.Tensor:
    """
    Replace one-offset diagonal entries with ideal bond lengths
    """
    mat_len = dist_mat.shape[1]
    bond_lengths = torch.cat([bond_lengths] * (mat_len // 3))[:mat_len - 1]

    dist_mat[1:, :-1][torch.eye(mat_len - 1) == 1] = bond_lengths
    dist_mat[:-1, 1:][torch.eye(mat_len - 1) == 1] = bond_lengths

    # Set chain break distance to arbitrarily large value for replacement by F-W algorithm
    if delim is not None:
        dist_mat[delim * 3 + 2, (delim + 1) * 3] = delim_value
        dist_mat[(delim + 1) * 3, delim * 3 + 2] = delim_value

    return dist_mat


def fill_dist_mat(dist_mat: torch.Tensor) -> torch.Tensor:
    """
    Fill sparse distance matrix using Floyd-Warshall shortest path algorithm
    """
    dist_mat[dist_mat != dist_mat] = ARBITRARILY_LARGE_VALUE
    for m in range(dist_mat.shape[0]):
        o = dist_mat[m]
        dist_mat = torch.min(torch.stack(
            [dist_mat, o.unsqueeze(0) + o.unsqueeze(1)]),
                             dim=0)[0]

    return dist_mat


def get_full_dist_mat(dist: torch.Tensor,
                      omega: torch.Tensor,
                      theta: torch.Tensor,
                      phi: torch.Tensor,
                      delim: int = None,
                      mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute full backbone atom distance atom distance matrix using
    dist, omega, theta, phi values by multi-dimensional scaling
    """

    # Ideal bond lengths and angles (from SO's MDS code)
    d = {
        "NA": torch.tensor(1.458),
        "AN": torch.tensor(1.458),
        "AC": torch.tensor(1.523),
        "CA": torch.tensor(1.523),
        "AB": torch.tensor(1.522),
        "BA": torch.tensor(1.522),
        "C_N": torch.tensor(1.329),
        "NB": torch.tensor(2.447),
        "BN": torch.tensor(2.447),
        "CB": torch.tensor(2.499),
        "BC": torch.tensor(2.499),
        "NC": torch.tensor(2.460),
        "CN": torch.tensor(2.460),
        "ANC": torch.tensor(0.615),
        "BANC": torch.tensor(-2.143),
        "NAB": torch.tensor(1.927),
        "BAN": torch.tensor(1.927),
        "NABB": theta.flatten().unsqueeze(-1),
        "BBAN": theta.transpose(0, 1).flatten().unsqueeze(-1),
        "ABB": phi.flatten().unsqueeze(-1),
        "BBA": phi.transpose(0, 1).flatten().unsqueeze(-1),
        "BB": dist.flatten().unsqueeze(-1),
        "ABBA": omega.flatten().unsqueeze(-1)
    }
    bond_lengths = torch.tensor([d["NA"], d["AC"], d["C_N"]])

    atoms = ["N", "A", "C"]
    atoms_len = len(atoms)

    seq_len = dist.shape[0]

    # Create ideal local coordinate system
    x, y = {}, {}
    x["N"] = torch.tensor([[0, 0, 0]]).float()
    x["A"] = torch.tensor([[0, 0, d["NA"]]]).float()
    x["B"] = torch.tensor([[
        0, d["AB"] * torch.sin(d["NAB"]),
        d["NA"] - d["AB"] * torch.cos(d["NAB"]).float()
    ]])
    x["C"] = place_fourth_atom(x["B"], x["A"], x["N"], d["NC"], d["ANC"],
                               d["BANC"])

    # Compute inter-residue atom positions
    y["B"] = place_fourth_atom(x["N"], x["A"], x["B"], d["BB"], d["ABB"],
                               d["NABB"])
    y["A"] = place_fourth_atom(x["A"], x["B"], y["B"], d["BA"], d["BBA"],
                               d["ABBA"])
    y["N"] = place_fourth_atom(x["B"], y["B"], y["A"], d["AN"], d["BAN"],
                               d["BBAN"])
    y["C"] = place_fourth_atom(y["B"], y["A"], y["N"], d["NC"], d["ANC"],
                               d["BANC"])

    # Create initial backbone distance matrix from computed coords
    dist_mat = []
    for atom_i in atoms:
        for atom_j in atoms:
            dist_mat_ij = (x[atom_i] - y[atom_j]).norm(dim=-1).reshape(
                seq_len, seq_len)

            if atom_i == atom_j:
                dist_mat_ij[torch.eye(seq_len) == 1] = 0
            else:
                dist_mat_ij[torch.eye(seq_len) == 1] = d[atom_i + atom_j]

            dist_mat.append(dist_mat_ij)

    raw_dist_mat = torch.stack(dist_mat).reshape(
        (atoms_len, atoms_len, seq_len, seq_len)).permute(
            (2, 0, 3, 1)).reshape((atoms_len * seq_len, atoms_len * seq_len))

    # Create sparse distance matrix, with potentially missing values
    sparse_dist_mat = raw_dist_mat.clone()
    if mask is not None:
        # Set masked positions to arbitrarily large value for replacement by F-W algorithm
        sparse_dist_mat[mask == 0] = ARBITRARILY_LARGE_VALUE

    # Fix bond lengths in sparse distance matrix
    sparse_dist_mat = fix_bond_lengths(sparse_dist_mat,
                                       bond_lengths,
                                       delim=delim)

    # Compute complete distance matrix
    full_dist_mat = fill_dist_mat(sparse_dist_mat)
    if delim is not None:
        # Store computed chain break distance
        delim_dist = full_dist_mat[delim * 3 + 2, (delim + 1) * 3].item()
    else:
        delim_dist = ARBITRARILY_LARGE_VALUE

    # Fix bond lengths
    full_dist_mat = fix_bond_lengths(full_dist_mat,
                                     bond_lengths,
                                     delim=delim,
                                     delim_value=delim_dist)

    # Symmetrize backbone distance matrix
    full_dist_mat = (full_dist_mat + full_dist_mat.transpose(0, 1)) / 2

    return full_dist_mat


def metric_MDS(dist_mat: torch.Tensor) -> torch.Tensor:
    """
    Find coords satisfying distance matrix via multi-dimensional scaling
    """
    mds = MDS(3, max_iter=500, dissimilarity="precomputed")
    coords = torch.tensor(mds.fit_transform(dist_mat))

    return coords


def generate_mds_coords(dist: torch.Tensor,
                        omega: torch.Tensor,
                        theta: torch.Tensor,
                        phi: torch.Tensor,
                        delim: int = None,
                        mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute set of N, CA, C, O, CB coords from dist, omega, theta, phi via multi-dimensional scaling
    """

    # Expand mask to cover all backbone atoms
    if mask is not None:
        mask = mask.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1)

    # Compute backbone distance matrix from dist, omega, theta, phi values via MDS
    full_dist_mat = get_full_dist_mat(dist,
                                      omega,
                                      theta,
                                      phi,
                                      delim=delim,
                                      mask=mask)
    ca_dist_mat = full_dist_mat[1::3, :][:, 1::3]

    full_coords = metric_MDS(full_dist_mat)
    full_coords = fix_chirality(full_coords)

    # Extract backbone atoms and place ideal CB and O atoms
    seq_len = dist.shape[0]
    n_coords, ca_coords, c_coords = full_coords.reshape(seq_len, -1,
                                                        3).permute(1, 0, 2)
    cb_coords = place_fourth_atom(c_coords, n_coords, ca_coords,
                                  torch.tensor(1.522), torch.tensor(1.927),
                                  torch.tensor(-2.143))
    o_coords = place_fourth_atom(torch.roll(n_coords, -1, 0), ca_coords,
                                 c_coords, torch.tensor(1.231),
                                 torch.tensor(2.108), torch.tensor(-3.142))

    full_coords = torch.stack(
        [n_coords, ca_coords, c_coords, o_coords, cb_coords], dim=1)

    return full_coords, ca_dist_mat


def save_PDB(out_pdb: str,
             coords: torch.Tensor,
             dist_mat: torch.Tensor,
             seq: str,
             delim: int = None) -> None:
    """
    Write set of N, CA, C, O, CB coords to PDB file
    """

    if type(delim) == type(None):
        delim = -1

    atoms = ['N', 'CA', 'C', 'O', 'CB']
    error = (dist_mat.fill_diagonal_(0) - calc_dist_mat(
        coords[:, 1], coords[:, 1]).float()).norm(dim=-1) / np.sqrt(len(seq))

    with open(out_pdb, "w") as f:
        k = 0
        for r, residue in enumerate(coords):
            AA = _aa_1_3_dict[seq[r]]
            for a, atom in enumerate(residue):
                if AA == "GLY" and atoms[a] == "CB": continue
                x, y, z = atom
                f.write(
                    "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                    % (k + 1, atoms[a], AA, "H" if r <= delim else "L", r + 1,
                       x, y, z, 1, error[r]))
                k += 1
        f.close()


def build_fv_mds(fasta_file: str,
                 out_pdb: str,
                 dist: torch.Tensor,
                 omega: torch.Tensor,
                 theta: torch.Tensor,
                 phi: torch.Tensor,
                 mask: torch.Tensor = None) -> None:
    """
    Generate atom coords from dist, omega, theta, phi tensors and write to PDB file
    """

    heavy_len = get_heavy_seq_len(fasta_file)
    seq = load_full_seq(fasta_file)

    mds_coords, ca_dist_mat = generate_mds_coords(dist,
                                                  omega,
                                                  theta,
                                                  phi,
                                                  delim=(heavy_len - 1),
                                                  mask=mask)

    save_PDB(out_pdb, mds_coords, ca_dist_mat, seq, delim=(heavy_len - 1))