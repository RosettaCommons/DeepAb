import time
import sys
import io
import requests
from os.path import splitext, basename
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
from Bio import SeqIO
from bisect import bisect_left, bisect_right
import torch
import numpy as np

from deepab.util.geometry import get_masked_mat, calc_dist_mat, calc_dihedral, calc_planar, place_fourth_atom
from deepab.util.masking import make_square_mask, MASK_VALUE
from deepab.util.util import get_fasta_chain_seq


def renumber_pdb(old_pdb, renum_pdb):
    success = False
    time.sleep(5)
    for i in range(10):
        try:
            with open(old_pdb, 'rb') as f:
                response = requests.post(
                    'http://www.bioinf.org.uk/abs/abnum/abnumpdb.cgi',
                    params={
                        "plain": "1",
                        "output": "-HL",
                        "scheme": "-c"
                    },
                    files={"pdb": f})

            success = response.status_code == 200 and not ("<html>"
                                                           in response.text)

            if success:
                break
            else:
                time.sleep((i + 1) * 5)
        except requests.exceptions.ConnectionError:
            time.sleep(60)

    # if success:
    if success:
        new_pdb_data = response.text
        with open(renum_pdb, "w") as f:
            f.write(new_pdb_data)
    else:
        print(
            "Failed to renumber PDB. This is likely due to a connection error or a timeout with the AbNum server."
        )


def pdb2fasta(pdb_file, num_chains=None):
    """Converts a PDB file to a fasta formatted string using its ATOM data"""
    pdb_id = basename(pdb_file).split('.')[0]
    parser = PDBParser()
    structure = parser.get_structure(pdb_id, pdb_file)

    real_num_chains = len([0 for _ in structure.get_chains()])
    if num_chains is not None and num_chains != real_num_chains:
        print('WARNING: Skipping {}. Expected {} chains, got {}'.format(
            pdb_file, num_chains, real_num_chains))
        return ''

    fasta = ''
    for chain in structure.get_chains():
        id_ = chain.id
        seq = seq1(''.join([residue.resname for residue in chain]))
        fasta += '>{}:{}\t{}\n'.format(pdb_id, id_, len(seq))
        max_line_length = 80
        for i in range(0, len(seq), max_line_length):
            fasta += f'{seq[i:i + max_line_length]}\n'
    return fasta


def get_atom_coord(residue, atom_type):
    if atom_type in residue:
        return residue[atom_type].get_coord()
    else:
        return [0, 0, 0]


def get_cb_or_ca_coord(residue):
    if 'CB' in residue:
        return residue['CB'].get_coord()
    elif 'CA' in residue:
        return residue['CA'].get_coord()
    else:
        return [0, 0, 0]


def get_continuous_ranges(residues):
    """ Returns ranges of residues which are continuously connected (peptide bond length 1.2-1.45 Å) """
    dists = []
    for res_i in range(len(residues) - 1):
        dists.append(
            np.linalg.norm(
                np.array(get_atom_coord(residues[res_i], "C")) -
                np.array(get_atom_coord(residues[res_i + 1], "N"))))

    ranges = []
    start_i = 0
    for d_i, d in enumerate(dists):
        if d > 1.45 or d < 1.2:
            ranges.append((start_i, d_i + 1))
            start_i = d_i + 1
        if d_i == len(dists) - 1:
            ranges.append((start_i, None))

    return ranges


def place_missing_cb_o(atom_coords):
    cb_coords = place_fourth_atom(atom_coords['C'], atom_coords['N'],
                                  atom_coords['CA'], torch.tensor(1.522),
                                  torch.tensor(1.927), torch.tensor(-2.143))
    o_coords = place_fourth_atom(
        torch.roll(atom_coords['N'], shifts=-1, dims=0), atom_coords['CA'],
        atom_coords['C'], torch.tensor(1.231), torch.tensor(2.108),
        torch.tensor(-3.142))

    bb_mask = get_atom_coords_mask(atom_coords['N']) & get_atom_coords_mask(
        atom_coords['CA']) & get_atom_coords_mask(atom_coords['C'])
    missing_cb = (get_atom_coords_mask(atom_coords['CB']) & bb_mask) == 0
    atom_coords['CB'][missing_cb] = cb_coords[missing_cb]

    bb_mask = get_atom_coords_mask(
        torch.roll(
            atom_coords['N'], shifts=-1, dims=0)) & get_atom_coords_mask(
                atom_coords['CA']) & get_atom_coords_mask(atom_coords['C'])
    missing_o = (get_atom_coords_mask(atom_coords['O']) & bb_mask) == 0
    atom_coords['O'][missing_o] = o_coords[missing_o]


def get_atom_coords(pdb_file, fasta_file=None):
    p = PDBParser()
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    if fasta_file:
        residues = []
        for chain in structure.get_chains():
            pdb_seq = get_pdb_chain_seq(pdb_file, chain.id)
            fasta_seq = get_fasta_chain_seq(fasta_file, chain.id)

            chain_residues = list(chain.get_residues())
            continuous_ranges = get_continuous_ranges(chain_residues)

            fasta_residues = [[]] * len(fasta_seq)
            fasta_r = (0, 0)
            for pdb_r in continuous_ranges:
                fasta_r_start = fasta_seq[fasta_r[1]:].index(
                    pdb_seq[pdb_r[0]:pdb_r[1]]) + fasta_r[1]
                fasta_r_end = (len(pdb_seq) if pdb_r[1] == None else
                               pdb_r[1]) - pdb_r[0] + fasta_r_start
                fasta_r = (fasta_r_start, fasta_r_end)
                fasta_residues[fasta_r[0]:fasta_r[1]] = chain_residues[
                    pdb_r[0]:pdb_r[1]]

            residues += fasta_residues
    else:
        residues = [r for r in structure.get_residues()]

    n_coords = torch.tensor([get_atom_coord(r, 'N') for r in residues])
    ca_coords = torch.tensor([get_atom_coord(r, 'CA') for r in residues])
    c_coords = torch.tensor([get_atom_coord(r, 'C') for r in residues])
    cb_coords = torch.tensor([get_atom_coord(r, 'CB') for r in residues])
    cb_ca_coords = torch.tensor([get_cb_or_ca_coord(r) for r in residues])
    o_coords = torch.tensor([get_atom_coord(r, 'O') for r in residues])

    atom_coords = {}
    atom_coords['N'] = n_coords
    atom_coords['CA'] = ca_coords
    atom_coords['C'] = c_coords
    atom_coords['CB'] = cb_coords
    atom_coords['CBCA'] = cb_ca_coords
    atom_coords['O'] = o_coords

    place_missing_cb_o(atom_coords)

    return atom_coords


def get_atom_coords_mask(coords):
    mask = torch.ByteTensor([1 if sum(_) != 0 else 0 for _ in coords])
    mask = mask & (1 - torch.any(torch.isnan(coords), dim=1).byte())
    return mask


def get_atom_coords_mask_for_dict(atom_coords):
    atom_coords_masks = {}
    for atom, coords in atom_coords.items():
        atom_coords_masks[atom] = get_atom_coords_mask(coords)

    return atom_coords_masks


def protein_pairwise_geometry_matrix(pdb_file,
                                     fasta_file=None,
                                     mask=None,
                                     mask_fill_value=MASK_VALUE,
                                     convert_to_degree=True,
                                     device=None):
    atom_coords = get_atom_coords(pdb_file, fasta_file=fasta_file)

    n_coords = atom_coords['N']
    ca_coords = atom_coords['CA']
    cb_coords = atom_coords['CB']
    o_coords = atom_coords['O']

    n_mask = get_atom_coords_mask(n_coords)
    ca_mask = get_atom_coords_mask(ca_coords)
    cb_mask = get_atom_coords_mask(cb_coords)
    o_mask = get_atom_coords_mask(o_coords)

    seq_len = len(ca_coords)
    if mask is None:
        mask = torch.ones(seq_len).byte()

    ca_dist_mat = get_masked_mat(calc_dist_mat(ca_coords, ca_coords),
                                 mask=make_square_mask(mask & ca_mask),
                                 mask_fill_value=mask_fill_value,
                                 device=device)
    cb_dist_mat = get_masked_mat(calc_dist_mat(cb_coords, cb_coords),
                                 mask=make_square_mask(mask & cb_mask),
                                 mask_fill_value=mask_fill_value,
                                 device=device)
    no_dist_mat = get_masked_mat(calc_dist_mat(n_coords, o_coords),
                                 mask=make_square_mask(mask & n_mask & o_mask),
                                 mask_fill_value=mask_fill_value,
                                 device=device)
    omega_mat = get_masked_mat(calc_dihedral(
        ca_coords.unsqueeze(1).expand(seq_len, seq_len, 3),
        cb_coords.unsqueeze(1).expand(seq_len, seq_len, 3),
        cb_coords.unsqueeze(0).expand(seq_len, seq_len, 3),
        ca_coords.unsqueeze(0).expand(seq_len, seq_len, 3),
        convert_to_degree=convert_to_degree),
                               mask=make_square_mask(mask & ca_mask & cb_mask),
                               mask_fill_value=mask_fill_value,
                               device=device)
    theta_mat = get_masked_mat(
        calc_dihedral(n_coords.unsqueeze(1).expand(seq_len, seq_len, 3),
                      ca_coords.unsqueeze(1).expand(seq_len, seq_len, 3),
                      cb_coords.unsqueeze(1).expand(seq_len, seq_len, 3),
                      cb_coords.unsqueeze(0).expand(seq_len, seq_len, 3),
                      convert_to_degree=convert_to_degree),
        mask=make_square_mask(mask & n_mask & ca_mask & cb_mask),
        mask_fill_value=mask_fill_value,
        device=device)
    phi_mat = get_masked_mat(calc_planar(
        ca_coords.unsqueeze(1).expand(seq_len, seq_len, 3),
        cb_coords.unsqueeze(1).expand(seq_len, seq_len, 3),
        cb_coords.unsqueeze(0).expand(seq_len, seq_len, 3),
        convert_to_degree=convert_to_degree),
                             mask=make_square_mask(mask & ca_mask & cb_mask),
                             mask_fill_value=mask_fill_value,
                             device=device)

    output_matrix = torch.stack(
        [ca_dist_mat, cb_dist_mat, no_dist_mat, omega_mat, theta_mat, phi_mat])

    return output_matrix


def get_pdb_chain_seq(pdb_file, chain_id):
    raw_fasta = pdb2fasta(pdb_file)
    fasta = SeqIO.parse(io.StringIO(raw_fasta), 'fasta')
    chain_sequences = {
        chain.id.split(':')[1]: str(chain.seq)
        for chain in fasta
    }
    if chain_id not in chain_sequences.keys():
        print(
            "No such chain in PDB file. Chain must have a chain ID of \"[PDB ID]:{}\""
            .format(chain_id))
        return None
    return chain_sequences[chain_id]


def heavy_chain_seq(pdb_file):
    return get_pdb_chain_seq(pdb_file, chain_id="H")


def cdr_indices(chothia_pdb_file, cdr):
    """Gets the index of a given CDR loop"""
    cdr_chothia_range_dict = {
        "h1": (26, 32),
        "h2": (52, 56),
        "h3": (95, 102),
        "l1": (24, 34),
        "l2": (50, 56),
        "l3": (89, 97)
    }

    cdr = str.lower(cdr)
    assert cdr in cdr_chothia_range_dict.keys()

    chothia_range = cdr_chothia_range_dict[cdr]
    chain_ids = []
    chain_id = cdr[0].upper()

    parser = PDBParser()
    pdb_id = basename(chothia_pdb_file).split('.')[0]
    structure = parser.get_structure(pdb_id, chothia_pdb_file)
    cdr_chain_structure = None
    for chain in structure.get_chains():
        chain_ids.append(chain.id)
        if chain.id == chain_id:
            cdr_chain_structure = chain
            break
    if cdr_chain_structure is None:
        print("PDB must have a chain with chain id \"[PBD ID]:{}\"".format(
            chain_id))
        sys.exit(-1)

    residue_id_nums = [res.get_id()[1] for res in cdr_chain_structure]

    # Binary search to find the start and end of the CDR loop
    cdr_start = bisect_left(residue_id_nums, chothia_range[0])
    cdr_end = bisect_right(residue_id_nums, chothia_range[1]) - 1

    if len(get_pdb_chain_seq(chothia_pdb_file,
                             chain_id=chain_id)) != len(residue_id_nums):
        print('ERROR in PDB file ' + chothia_pdb_file)
        print('residue id len', len(residue_id_nums))
        print('seq', len(heavy_chain_seq(chothia_pdb_file)))

    if chain_id == "L" and "H" in chain_ids:
        cdr_start += len(heavy_chain_seq(chothia_pdb_file))
        cdr_end += len(heavy_chain_seq(chothia_pdb_file))

    return cdr_start, cdr_end


def h3_indices(chothia_pdb_file):
    """Gets the index of the CDR H3 loop"""

    return cdr_indices(chothia_pdb_file, cdr="h3")


def get_chain_numbering(pdb_file, chain_id):
    seq = []
    parser = PDBParser()
    structure = parser.get_structure("_", pdb_file)
    for chain in structure.get_chains():
        if chain.id == chain_id:
            for r in chain.get_residues():
                res_num = str(r._id[1]) + r._id[2]
                res_num = res_num.replace(" ", "")
                seq.append(res_num)

            return seq


def write_pdb_bfactor(in_pdb_file, out_pdb_file, bfactor):
    parser = PDBParser()
    structure = parser.get_structure("_", in_pdb_file)

    i = 0
    for chain in structure.get_chains():
        for r in chain.get_residues():
            [a.set_bfactor(bfactor[i]) for a in r.get_atoms()]
            i += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_file)