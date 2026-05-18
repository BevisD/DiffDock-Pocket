import copy
import numpy as np
import torch
from torch_geometric.data import Batch

from utils.diffusion_utils import modify_conformer, set_time, modify_sidechains
from utils.potentials import get_potential_gradients


# ────────────────────────── small helpers ───────────────────────────────

def _to_data_list(data):
    """Coerce a PyG Batch (CPU path) or a Python list (GPU path) to a list of HeteroData."""
    if isinstance(data, list):
        return data
    return data.to_data_list()


def _dt_at(schedule, t_idx, K):
    return schedule[t_idx] - schedule[t_idx + 1] if t_idx < K - 1 else schedule[t_idx]


def _diff_coeffs(sigmas, model_args, factor_keys):
    tr_sigma, rot_sigma, tor_sigma, sc_tor_sigma = sigmas
    tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
    rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))
    tor_g = (tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
             if 'tor' in factor_keys else None)
    sc_g = (sc_tor_sigma * torch.sqrt(torch.tensor(2 * np.log(
        model_args.sidechain_tor_sigma_max / model_args.sidechain_tor_sigma_min)))
            if 'sc_tor' in factor_keys else None)
    return tr_g, rot_g, tor_g, sc_g


def _set_time_on_batch(data_list, t_idx, t_tr, t_rot, t_tor, t_sc,
                       t_schedule, model_args, async_, device):
    """Set time on the data and return the model's input:
       GPU: model is DataParallel-wrapped — return the list, set_time per graph.
       CPU: return a Batch with time already set on it."""
    t_norm = t_schedule[t_idx] if t_schedule is not None else None
    all_atoms = 'all_atoms' in model_args and model_args.all_atoms
    misc = (hasattr(model_args, 'include_miscellaneous_atoms')
            and model_args.include_miscellaneous_atoms)

    if device.type == 'cuda':
        for cg in data_list:
            set_time(cg, t_norm, t_tr, t_rot, t_tor, t_sc, 1,
                     all_atoms, async_, device,
                     include_miscellaneous_atoms=misc)
        return data_list

    batch = Batch.from_data_list(data_list).to(device)
    set_time(batch, t_norm, t_tr, t_rot, t_tor, t_sc, batch.num_graphs,
             all_atoms, async_, device,
             include_miscellaneous_atoms=misc)
    return batch


def _terminal_adjoint(data_list, device, factor_keys, energy_fn, tor_counts, sc_tor_counts):
    """ã_K = ∇E(X_K), per molecule. `get_potential_gradients` may return None — treat as zero
    with the right per-molecule shape."""
    tr_l, rot_l, tor_l, sc_l = [], [], [], []
    for i, cg in enumerate(data_list):
        tr_g_, rot_g_, tor_g_, sc_g_ = get_potential_gradients(cg, energy_fn)
        tr_l.append((tr_g_ if tr_g_ is not None else torch.zeros(3)).detach().to(device))
        rot_l.append((rot_g_ if rot_g_ is not None else torch.zeros(3)).detach().to(device))
        if 'tor' in factor_keys:
            g = tor_g_ if tor_g_ is not None else torch.zeros(tor_counts[i])
            tor_l.append(g.detach().to(device))
        if 'sc_tor' in factor_keys:
            g = sc_g_ if sc_g_ is not None else torch.zeros(sc_tor_counts[i])
            sc_l.append(g.detach().to(device))
    out = {'tr': torch.stack(tr_l, dim=0), 'rot': torch.stack(rot_l, dim=0)}
    if 'tor' in factor_keys: out['tor'] = torch.cat(tor_l, dim=0)
    if 'sc_tor' in factor_keys: out['sc_tor'] = torch.cat(sc_l, dim=0)
    return out


def _apply_xi(data_list, xi, tor_offsets, sc_tor_offsets, factor_keys, pivot):
    """Apply the zero-with-grad Lie-algebra perturbation to each complex_graph."""
    has_tor = 'tor' in factor_keys
    has_sc = 'sc_tor' in factor_keys
    for i, cg in enumerate(data_list):
        tor_slice = (xi['tor'][tor_offsets[i]:tor_offsets[i + 1]]
                     if has_tor and tor_offsets[i + 1] > tor_offsets[i] else None)
        modify_conformer(
            cg,
            xi['tr'][i:i + 1],
            xi['rot'][i],
            tor_slice,
            pivot=pivot,
        )
        if has_sc and sc_tor_offsets[i + 1] > sc_tor_offsets[i]:
            modify_sidechains(
                cg,
                xi['sc_tor'][sc_tor_offsets[i]:sc_tor_offsets[i + 1]],
            )


def _restore_positions(data_list, traj_t, sc_traj_t, flexible_sidechains, no_sc_in_batch, device):
    """traj_t and sc_traj_t are Python lists of per-molecule tensors."""
    for i, cg in enumerate(data_list):
        cg['ligand'].pos = traj_t[i].clone().to(device)
        if 'atom' in cg.node_types:
            cg['atom'].pos = cg['atom'].pos.detach().clone()
        if flexible_sidechains and not no_sc_in_batch:
            sub_idx = cg['flexResidues'].subcomponents.unique()
            cg['atom'].pos[sub_idx] = sc_traj_t[i].clone().to(device)


def _mean_energy(data_list, energy_fn):
    total = 0.0
    for cg in data_list:
        E = energy_fn(cg)
        total += E.detach().item() if torch.is_tensor(E) else float(E)
    return total / len(data_list)


# ────────────────────────────── main ────────────────────────────────────

def adjoint_loss(data_list, model_base, model_finetune, energy_fn, inference_steps,
                 tr_schedule, rot_schedule, tor_schedule,
                 sidechain_tor_schedule, device, t_to_sigma, model_args,
                 tr_weight=1, rot_weight=1, tor_weight=1, sc_tor_weight=1,
                 asyncronous_noise_schedule=False, t_schedule=None,
                 no_final_step_noise=False, pivot=None, flexible_sidechains=None):
    """
    Returns
    -------
    total_loss : scalar tensor, connected via autograd to model_finetune's params
    log_dict   : dict of detached scalars per factor + total, suitable for logging
    """
    flexible_sidechains = (model_args.flexible_sidechains if flexible_sidechains is None
                           else flexible_sidechains)

    # Normalise input: list-of-HeteroData regardless of CPU/GPU
    data_list = _to_data_list(data_list)

    if flexible_sidechains:
        no_sc_in_batch = sum([len(c["flexResidues"].subcomponents) for c in data_list]) == 0
        if no_sc_in_batch:
            data_list = copy.deepcopy(data_list)
            for c in data_list:
                del c["flexResidues"]
    else:
        no_sc_in_batch = True

    # Active factors — tr and rot always on
    factor_keys = ('tr', 'rot')
    if not model_args.no_torsion:
        factor_keys = factor_keys + ('tor',)
    if flexible_sidechains:
        factor_keys = factor_keys + ('sc_tor',)

    for p in model_base.parameters():
        p.requires_grad_(False)

    N, K = len(data_list), inference_steps

    # ── Per-molecule counts and cumulative offsets (variable across the batch) ──
    if 'tor' in factor_keys:
        tor_counts = [int(cg['ligand'].edge_mask.sum()) for cg in data_list]
    else:
        tor_counts = [0] * N
    tor_offsets = np.cumsum([0] + tor_counts).tolist()  # length N+1
    total_tor = tor_offsets[-1]

    if 'sc_tor' in factor_keys and not no_sc_in_batch:
        sc_tor_counts = [len(cg['flexResidues'].edge_idx) for cg in data_list]
    else:
        sc_tor_counts = [0] * N
    sc_tor_offsets = np.cumsum([0] + sc_tor_counts).tolist()
    total_sc_tor = sc_tor_offsets[-1]

    trajectory, sc_trajectory, g_trajectory = [], [], []

    # ── Rollout under FT-controlled SDE  (no autograd) ─────────────────
    with torch.no_grad():
        for t_idx in range(K):
            t_tr, t_rot, t_tor, t_sc = (tr_schedule[t_idx], rot_schedule[t_idx],
                                        tor_schedule[t_idx], sidechain_tor_schedule[t_idx])
            dt_tr = _dt_at(tr_schedule, t_idx, K)
            dt_rot = _dt_at(rot_schedule, t_idx, K)
            dt_tor = _dt_at(tor_schedule, t_idx, K)
            dt_sc = _dt_at(sidechain_tor_schedule, t_idx, K)

            # store positions as lists of per-molecule tensors (variable atom counts)
            trajectory.append([cg['ligand'].pos.clone() for cg in data_list])
            sc_trajectory.append(None if no_sc_in_batch else [
                cg['atom'].pos.clone()[cg['flexResidues'].subcomponents.unique()]
                for cg in data_list])

            sigmas = t_to_sigma(t_tr, t_rot, t_tor, t_sc)
            tr_g, rot_g, tor_g, sc_g = _diff_coeffs(sigmas, model_args, factor_keys)

            model_input = _set_time_on_batch(
                data_list, t_idx, t_tr, t_rot, t_tor, t_sc,
                t_schedule, model_args, asyncronous_noise_schedule, device)
            tr_score, rot_score, tor_score, sc_score = model_finetune(model_input)

            last = (t_idx == K - 1)
            tr_z = torch.zeros(N, 3) if (no_final_step_noise and last) else torch.randn(N, 3)
            rot_z = torch.zeros(N, 3) if (no_final_step_noise and last) else torch.randn(N, 3)
            tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z)
            rot_perturb = (rot_g ** 2 * dt_rot * rot_score.cpu() + rot_g * np.sqrt(dt_rot) * rot_z)

            if 'tor' in factor_keys:
                tor_z = (torch.zeros(tor_score.shape) if (no_final_step_noise and last)
                         else torch.randn(*tor_score.shape))
                tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu()
                               + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
            else:
                tor_perturb = None

            if 'sc_tor' in factor_keys:
                sc_z = (torch.zeros(sc_score.shape) if (no_final_step_noise and last)
                        else torch.randn(*sc_score.shape))
                sc_perturb = (sc_g ** 2 * dt_sc * sc_score.cpu()
                              + sc_g * np.sqrt(dt_sc) * sc_z).numpy()
            else:
                sc_perturb = None

            # Apply per-molecule sidechain perturbation (skip molecules with zero sc-torsions)
            if 'sc_tor' in factor_keys:
                for i, cg in enumerate(data_list):
                    if sc_tor_counts[i] > 0:
                        modify_sidechains(cg, sc_perturb[sc_tor_offsets[i]:sc_tor_offsets[i + 1]])

            data_list = [
                modify_conformer(
                    cg,
                    tr_perturb[i:i + 1],
                    rot_perturb[i:i + 1].squeeze(0),
                    (tor_perturb[tor_offsets[i]:tor_offsets[i + 1]]
                     if ('tor' in factor_keys and tor_counts[i] > 0) else None),
                    pivot=pivot,
                )
                for i, cg in enumerate(data_list)
            ]

            g_trajectory.append((tr_g, rot_g, tor_g, sc_g))

    # Terminal mean energy
    mean_energy = _mean_energy(data_list, energy_fn)

    # ── Terminal adjoint  ã_K  ──────────────────────────────────────────
    a = _terminal_adjoint(data_list, device, factor_keys, energy_fn=energy_fn,
                          tor_counts=tor_counts, sc_tor_counts=sc_tor_counts)

    # ── Backward walk + matching loss ───────────────────────────────────
    losses = {k: torch.zeros((), device=device) for k in factor_keys}
    base_losses = {k: torch.zeros((), device=device) for k in factor_keys}

    schedules = {'tr': tr_schedule, 'rot': rot_schedule,
                 'tor': tor_schedule, 'sc_tor': sidechain_tor_schedule}

    for t_idx in reversed(range(K)):
        t_tr, t_rot, t_tor, t_sc = (tr_schedule[t_idx], rot_schedule[t_idx],
                                    tor_schedule[t_idx], sidechain_tor_schedule[t_idx])
        dt = {k: _dt_at(schedules[k], t_idx, K) for k in factor_keys}

        raw_gs = g_trajectory[t_idx]
        g_dict = {}
        for k, g in zip(('tr', 'rot', 'tor', 'sc_tor'), raw_gs):
            if k not in factor_keys:
                continue
            g_dict[k] = g.to(device) if torch.is_tensor(g) else torch.as_tensor(g, device=device)

        _restore_positions(data_list, trajectory[t_idx], sc_trajectory[t_idx],
                           flexible_sidechains, no_sc_in_batch, device)

        xi = {
            'tr': torch.zeros(N, 3, device=device, requires_grad=True),
            'rot': torch.zeros(N, 3, device=device, requires_grad=True),
        }
        if 'tor' in factor_keys:
            xi['tor'] = torch.zeros(total_tor, device=device, requires_grad=True)
        if 'sc_tor' in factor_keys:
            xi['sc_tor'] = torch.zeros(total_sc_tor, device=device, requires_grad=True)

        _apply_xi(data_list, xi, tor_offsets, sc_tor_offsets, factor_keys, pivot)

        model_input = _set_time_on_batch(
            data_list, t_idx, t_tr, t_rot, t_tor, t_sc,
            t_schedule, model_args, asyncronous_noise_schedule, device)
        s_b = dict(zip(('tr', 'rot', 'tor', 'sc_tor'), model_base(model_input)))
        s_f = dict(zip(('tr', 'rot', 'tor', 'sc_tor'), model_finetune(model_input)))

        vjps = torch.autograd.grad(
            outputs=[s_b[k] for k in factor_keys],
            inputs=[xi[k] for k in factor_keys],
            grad_outputs=[(g_dict[k] ** 2 * dt[k]) * a[k] for k in factor_keys],
            retain_graph=True, create_graph=False, allow_unused=True,
        )
        vjp_dict = {k: (v if v is not None else torch.zeros_like(xi[k]))
                    for k, v in zip(factor_keys, vjps)}

        a = {k: (a[k] + vjp_dict[k]).detach() for k in factor_keys}

        for k in factor_keys:
            delta = s_f[k] - s_b[k].detach()
            weight = g_dict[k] ** 2 * dt[k]
            losses[k] = losses[k] + weight * ((delta + a[k]) ** 2).sum()
            base_losses[k] = base_losses[k] + weight * (a[k] ** 2).sum()

    weight_map = {'tr': tr_weight, 'rot': rot_weight, 'tor': tor_weight, 'sc_tor': sc_tor_weight}
    total_loss      = sum(weight_map[k] * losses[k]      for k in factor_keys) / N
    total_base_loss = sum(weight_map[k] * base_losses[k] for k in factor_keys) / N

    all_factor_keys = ('tr', 'rot', 'tor', 'sc_tor')
    log_dict = {}
    for k in all_factor_keys:
        if k in factor_keys:
            log_dict[f'{k}_loss']      = (losses[k].detach()      / N).cpu()
            log_dict[f'{k}_base_loss'] = (base_losses[k].detach() / N).cpu()
        else:
            log_dict[f'{k}_loss']      = torch.zeros(())
            log_dict[f'{k}_base_loss'] = torch.zeros(())
    log_dict['total_loss']      = total_loss.detach().cpu()
    log_dict['total_base_loss'] = total_base_loss.detach().cpu()

    return total_loss, log_dict
