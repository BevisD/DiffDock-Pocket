import torch
from rdkit.Chem import GetPeriodicTable

from datasets.steric_clash import VAN_DER_WAALS_RADII, OVERLAP_DISTANCE
from utils.diffusion_utils import modify_conformer, modify_sidechains

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
    device = atomic_numbers.device
    return VAN_DER_WAALS_RADII_LUT[atomic_numbers.cpu() - 1].to(device)


def get_steric_clash_overlap(pos_1, pos_2, atomic_numbers_1, atomic_numbers_2):
    assert pos_1.device == pos_2.device
    device = pos_1.device

    atomic_radii_1 = get_atomic_radii(atomic_numbers_1).unsqueeze(0).to(device)
    atomic_radii_2 = get_atomic_radii(atomic_numbers_2).unsqueeze(0).to(device)

    cross_distances = torch.cdist(pos_1, pos_2)
    ramanchandran_radii = atomic_radii_1[:, :, None] + atomic_radii_2[:, None, :] - 2 * OVERLAP_DISTANCE

    return torch.clamp(ramanchandran_radii - cross_distances, min=0.0)


class StericClashEnergy:
    def __init__(self, args):
        self.energy_weight = float(args.energy_weight)

    def __call__(self, complex_graph):
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

        return self.energy_weight * E



def get_potential_gradients(complex_graph, potential):
    device = complex_graph["ligand"].pos.device
    dtype = complex_graph["ligand"].pos.dtype

    num_torsions = len(complex_graph['ligand'].mask_rotate)
    num_sidechains = complex_graph['flexResidues'].num_nodes

    with torch.enable_grad():
        tr_update = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
        rot_update = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
        tor_update = torch.zeros(num_torsions, device=device, dtype=dtype, requires_grad=True)
        sidechain_tor_update = torch.zeros(num_sidechains, device=device, dtype=dtype, requires_grad=True)

        complex_graph["ligand"].pos = complex_graph["ligand"].pos.detach()
        complex_graph["atom"].pos = complex_graph["atom"].pos.detach()

        modify_conformer(complex_graph, tr_update, rot_update, tor_update)
        modify_sidechains(complex_graph, sidechain_tor_update)

        pot = potential(complex_graph)

        grads = torch.autograd.grad(
            pot,
            [tr_update, rot_update, tor_update, sidechain_tor_update],
            allow_unused=True,
        )

        grads = tuple(
            g.detach() if g is not None else None
            for g in grads
        )

    return grads


def get_energy_function(name, args):
    registry = {
        "steric_clash": StericClashEnergy,
    }

    if name not in registry:
        raise ValueError(f"Unknown energy function: {name}")

    return registry[name](args)
