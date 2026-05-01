import torch
from rdkit.Chem import GetPeriodicTable

from datasets.steric_clash import VAN_DER_WAALS_RADII, OVERLAP_DISTANCE

PERIODIC_TABLE = GetPeriodicTable()

VAN_DER_WAALS_RADII_LUT = torch.full((118,), VAN_DER_WAALS_RADII["default"])
for sym, rad in VAN_DER_WAALS_RADII.items():
    if sym == "default":
        continue

    z = PERIODIC_TABLE.GetAtomicNumber(sym)
    VAN_DER_WAALS_RADII_LUT[z - 1] = rad


def get_ligand_atomic_numbers(complex_graph):
    return complex_graph["ligand"].x[:, 0] + 1


def get_rec_atomic_numbers(complex_graph):
    return complex_graph["atom"].x[:, 1] + 1


def get_atomic_radii(atomic_numbers):
    return VAN_DER_WAALS_RADII_LUT[atomic_numbers - 1]


def get_steric_clash_overlap(pos_1, pos_2, atomic_numbers_1, atomic_numbers_2):
    atomic_radii_1 = get_atomic_radii(atomic_numbers_1).unsqueeze(0)
    atomic_radii_2 = get_atomic_radii(atomic_numbers_2).unsqueeze(0)

    cross_distances = torch.cdist(pos_1, pos_2)
    ramanchandran_radii = atomic_radii_1[:, :, None] + atomic_radii_2[:, None, :] - 2 * OVERLAP_DISTANCE

    return torch.clamp(ramanchandran_radii - cross_distances, min=0.0)


def get_steric_clash_energy(complex_graph):
    lig_pos = complex_graph["ligand"].pos
    rec_pos = complex_graph["atom"].pos

    lig_z = get_ligand_atomic_numbers(complex_graph)
    rec_z = get_rec_atomic_numbers(complex_graph)

    E = torch.zeros((), device=lig_pos.device, dtype=lig_pos.dtype)

    lig_lig_overlap = get_steric_clash_overlap(
        lig_pos[None, :],
        lig_pos[None, :],
        lig_z,
        lig_z,
    )
    E += (lig_lig_overlap ** 2).sum()

    lig_rec_overlap = get_steric_clash_overlap(
        lig_pos[None, :],
        rec_pos[None, :],
        lig_z,
        rec_z,
    )
    E += (lig_rec_overlap ** 2).sum()

    return E
