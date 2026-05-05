import copy
import numpy as np
import torch
from torch_geometric.data import Batch

from utils.diffusion_utils import modify_conformer, set_time, modify_sidechains
from utils.potentials import get_steric_clash_energy, get_potential_gradients


# ────────────────────────── small helpers ───────────────────────────────

def _dt_at(schedule, t_idx, K):
    return schedule[t_idx] - schedule[t_idx + 1] if t_idx < K - 1 else schedule[t_idx]


def _diff_coeffs(sigmas, model_args, flexible_sidechains):
    """Reverse-SDE diffusion coefficients g(t) for each factor."""
    tr_sigma, rot_sigma, tor_sigma, sc_tor_sigma = sigmas
    tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
    rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))
    tor_g = (tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
             if not model_args.no_torsion else None)
    sc_g = (sc_tor_sigma * torch.sqrt(torch.tensor(2 * np.log(
        model_args.sidechain_tor_sigma_max / model_args.sidechain_tor_sigma_min)))
            if flexible_sidechains else None)
    return tr_g, rot_g, tor_g, sc_g


def _set_time_on_batch(batch, t_idx, t_tr, t_rot, t_tor, t_sc, t_schedule, model_args, async_, device):
    set_time(batch,
             t_schedule[t_idx] if t_schedule is not None else None,
             t_tr, t_rot, t_tor, t_sc, batch.num_graphs,
             'all_atoms' in model_args and model_args.all_atoms,
             async_, device,
             include_miscellaneous_atoms=hasattr(model_args, 'include_miscellaneous_atoms')
                                         and model_args.include_miscellaneous_atoms)


def _terminal_adjoint(data_list, device, flexible_sidechains, energy_fn=get_steric_clash_energy):
    """ã_K = ∇E(X_K), per molecule, stacked / concatenated."""
    tr_l, rot_l, tor_l, sc_l = [], [], [], []
    for cg in data_list:
        tr_g_, rot_g_, tor_g_, sc_g_ = get_potential_gradients(cg, energy_fn)
        tr_l.append(tr_g_.detach().to(device))
        rot_l.append(rot_g_.detach().to(device))
        tor_l.append(tor_g_.detach().to(device))
        if flexible_sidechains:
            sc_l.append(sc_g_.detach().to(device))
    return {
        'tr': torch.stack(tr_l, dim=0),  # (N, 3)
        'rot': torch.stack(rot_l, dim=0),  # (N, 3)
        'tor': torch.cat(tor_l, dim=0),  # (N * tor_per_mol,)
        'sc_tor': torch.cat(sc_l, dim=0) if flexible_sidechains else None,
    }


def _apply_xi(data_list, xi, tor_per_mol, sc_tor_per_mol, flexible_sidechains, pivot):
    """Apply the zero-with-grad Lie-algebra perturbation to each complex_graph."""
    for i, cg in enumerate(data_list):
        modify_conformer(
            cg,
            xi['tr'][i:i + 1],
            xi['rot'][i],
            xi['tor'][i * tor_per_mol:(i + 1) * tor_per_mol],
            pivot=pivot,
        )
        if flexible_sidechains:
            modify_sidechains(
                cg,
                xi['sc_tor'][i * sc_tor_per_mol:(i + 1) * sc_tor_per_mol],
            )


def _restore_positions(data_list, traj_t, sc_traj_t, flexible_sidechains, no_sc_in_batch, device):
    for i, cg in enumerate(data_list):
        cg['ligand'].pos = traj_t[i].clone().to(device)
        if flexible_sidechains and not no_sc_in_batch:
            sub_idx = cg['flexResidues'].subcomponents.unique()
            cg['atom'].pos[sub_idx] = sc_traj_t[i].clone().to(device)


# ────────────────────────────── main ────────────────────────────────────

def adjoint_loss(data_list, model_base, model_finetune, inference_steps,
                 tr_schedule, rot_schedule, tor_schedule,
                 sidechain_tor_schedule, device, t_to_sigma, model_args,
                 tr_weight=1, rot_weight=1, tor_weight=1, sc_tor_weight=1,
                 asyncronous_noise_schedule=False, t_schedule=None,
                 no_final_step_noise=False, pivot=None, flexible_sidechains=None,
                 energy_fn=get_steric_clash_energy):
    """
    Returns
    -------
    total_loss : scalar tensor, connected via autograd to model_finetune's params
    log_dict   : dict of detached scalars per factor + total, suitable for logging
    """
    flexible_sidechains = (model_args.flexible_sidechains if flexible_sidechains is None
                           else flexible_sidechains)

    if flexible_sidechains:
        no_sc_in_batch = sum([len(c["flexResidues"].subcomponents) for c in data_list]) == 0
        if no_sc_in_batch:
            data_list = copy.deepcopy(data_list)
            for c in data_list:
                del c["flexResidues"]
    else:
        no_sc_in_batch = True

    for p in model_base.parameters():  # ensure base is frozen
        p.requires_grad_(False)

    N, K = len(data_list), inference_steps
    trajectory, sc_trajectory, g_trajectory = [], [], []
    tor_per_mol = sc_tor_per_mol = None

    # ── Rollout under the FT-controlled SDE  (no autograd) ─────────────
    with torch.no_grad():
        for t_idx in range(K):
            t_tr, t_rot, t_tor, t_sc = (tr_schedule[t_idx], rot_schedule[t_idx],
                                        tor_schedule[t_idx], sidechain_tor_schedule[t_idx])
            dt_tr = _dt_at(tr_schedule, t_idx, K)
            dt_rot = _dt_at(rot_schedule, t_idx, K)
            dt_tor = _dt_at(tor_schedule, t_idx, K)
            dt_sc = _dt_at(sidechain_tor_schedule, t_idx, K)

            trajectory.append(torch.stack(
                [cg['ligand'].pos.clone() for cg in data_list], dim=0))
            sc_trajectory.append(None if no_sc_in_batch else torch.stack(
                [cg['atom'].pos.clone()[cg['flexResidues'].subcomponents.unique()]
                 for cg in data_list], dim=0))

            sigmas = t_to_sigma(t_tr, t_rot, t_tor, t_sc)
            tr_g, rot_g, tor_g, sc_g = _diff_coeffs(sigmas, model_args, flexible_sidechains)

            batch = Batch.from_data_list(data_list).to(device)
            _set_time_on_batch(batch, t_idx, t_tr, t_rot, t_tor, t_sc, t_schedule,
                               model_args, asyncronous_noise_schedule, device)
            tr_score, rot_score, tor_score, sc_score = model_finetune(batch)

            last = (t_idx == K - 1)
            tr_z = torch.zeros(N, 3) if (no_final_step_noise and last) else torch.randn(N, 3)
            rot_z = torch.zeros(N, 3) if (no_final_step_noise and last) else torch.randn(N, 3)
            tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z)
            rot_perturb = (rot_g ** 2 * dt_rot * rot_score.cpu() + rot_g * np.sqrt(dt_rot) * rot_z)

            if model_args.no_torsion:
                raise NotImplementedError("No-torsion not implemented yet for adjoint matching")
            tor_z = torch.zeros(tor_score.shape) if (no_final_step_noise and last) else torch.randn(*tor_score.shape)
            tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
            tor_per_mol = tor_perturb.shape[0] // N

            if not flexible_sidechains:
                raise NotImplementedError("No-sidechain-torsion not implemented yet for adjoint matching")
            sc_z = torch.zeros(sc_score.shape) if (no_final_step_noise and last) else torch.randn(*sc_score.shape)
            sc_perturb = (sc_g ** 2 * dt_sc * sc_score.cpu() + sc_g * np.sqrt(dt_sc) * sc_z).numpy()
            sc_tor_per_mol = sc_perturb.shape[0] // N

            for i, cg in enumerate(data_list):
                modify_sidechains(cg, sc_perturb[i * sc_tor_per_mol:(i + 1) * sc_tor_per_mol])
            data_list = [
                modify_conformer(cg,
                                 tr_perturb[i:i + 1],
                                 rot_perturb[i:i + 1].squeeze(0),
                                 tor_perturb[i * tor_per_mol:(i + 1) * tor_per_mol],
                                 pivot=pivot)
                for i, cg in enumerate(data_list)
            ]

            g_trajectory.append((tr_g, rot_g, tor_g, sc_g))

    # ── Terminal adjoint  ã_K  ──────────────────────────────────────────
    a = _terminal_adjoint(data_list, device, flexible_sidechains, energy_fn=energy_fn)

    # ── Backward walk + matching loss ───────────────────────────────────
    factor_keys = ('tr', 'rot', 'tor') + (('sc_tor',) if flexible_sidechains else ())
    losses = {k: torch.zeros((), device=device) for k in factor_keys}
    base_losses = {k: torch.zeros((), device=device) for k in factor_keys}

    schedules = {
        'tr': tr_schedule, 'rot': rot_schedule, 'tor': tor_schedule, 'sc_tor': sidechain_tor_schedule,
    }

    for t_idx in reversed(range(K)):
        t_tr, t_rot, t_tor, t_sc = (tr_schedule[t_idx], rot_schedule[t_idx],
                                    tor_schedule[t_idx], sidechain_tor_schedule[t_idx])
        dt = {k: _dt_at(schedules[k], t_idx, K) for k in factor_keys}

        gs = [g.to(device) if torch.is_tensor(g) else torch.as_tensor(g, device=device)
              for g in g_trajectory[t_idx]]
        g_dict = dict(zip(('tr', 'rot', 'tor', 'sc_tor'), gs))

        # Restore positions to X_k
        _restore_positions(data_list, trajectory[t_idx], sc_trajectory[t_idx],
                           flexible_sidechains, no_sc_in_batch, device)

        # Build xi (zero, requires_grad=True)
        xi = {
            'tr': torch.zeros(N, 3, device=device, requires_grad=True),
            'rot': torch.zeros(N, 3, device=device, requires_grad=True),
            'tor': torch.zeros(N * tor_per_mol, device=device, requires_grad=True),
        }
        if flexible_sidechains:
            xi['sc_tor'] = torch.zeros(N * sc_tor_per_mol, device=device, requires_grad=True)

        _apply_xi(data_list, xi, tor_per_mol, sc_tor_per_mol, flexible_sidechains, pivot)

        # One forward each through base and FT (positions now depend on xi)
        batch = Batch.from_data_list(data_list).to(device)
        _set_time_on_batch(batch, t_idx, t_tr, t_rot, t_tor, t_sc, t_schedule,
                           model_args, asyncronous_noise_schedule, device)
        s_b = dict(zip(('tr', 'rot', 'tor', 'sc_tor'), model_base(batch)))
        s_f = dict(zip(('tr', 'rot', 'tor', 'sc_tor'), model_finetune(batch)))

        # VJP of s_base wrt xi with v = g²·Δt·a_{k+1}
        vjps = torch.autograd.grad(
            outputs=[s_b[k] for k in factor_keys],
            inputs=[xi[k] for k in factor_keys],
            grad_outputs=[(g_dict[k] ** 2 * dt[k]) * a[k] for k in factor_keys],
            retain_graph=True, create_graph=False, allow_unused=True,
        )
        vjp_dict = dict(zip(factor_keys, vjps))

        # Adjoint update — vector addition in each Lie-algebra dual
        a = {k: (a[k] + vjp_dict[k]).detach() for k in factor_keys}

        # Matching loss term  g²·Δt·||Δs + a_k||²
        for k in factor_keys:
            delta = s_f[k] - s_b[k].detach()
            weight = g_dict[k] ** 2 * dt[k]
            losses[k] = losses[k] + weight * ((delta + a[k]) ** 2).sum()
            base_losses[k] = base_losses[k] + weight * (a[k] ** 2).sum()

    weights = dict(zip(factor_keys, (tr_weight, rot_weight, tor_weight, sc_tor_weight)))
    total_loss = sum(weights[k] * losses[k] for k in factor_keys) / N
    log_dict = {f'{k}_loss': (losses[k].detach() / N).item() for k in factor_keys}
    log_dict.update(
        {f'{k}_base_loss': (base_losses[k].detach() / N).item() for k in factor_keys})
    log_dict['total_loss'] = total_loss.detach().item()
    log_dict['total_base_loss'] = (sum(base_losses.values()).detach() / N).item()
    # log_dict['mean_energy']     = mean_energy
    return total_loss, log_dict
