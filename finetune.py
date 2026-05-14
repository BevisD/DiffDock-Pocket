import copy
import math
import os
from functools import partial
from argparse import Namespace

import torch
from utils.potentials import get_energy_function

torch.multiprocessing.set_sharing_strategy('file_system')

if os.name != 'nt':
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

    # Only raise if safe
    if hard >= 64000:
        target = 64000
    else:
        target = hard

    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

import yaml

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, t_to_sigma_individual
from datasets.pdbbind import construct_loader
from utils.adjoint import adjoint_loss
from utils.parsing import parse_train_args
from utils.training import test_epoch, inference_epoch_fix
from utils.finetuning import finetune_epoch
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage, \
    get_default_device


def finetune(args, model_base, model_finetune, energy_fn, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma,
             run_dir):
    device = args.device
    best_val_loss = math.inf
    best_val_sc_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_val_inference_sc_value = 0
    best_val_inference_steric_clashes_value = math.inf
    best_epoch = 0
    best_val_inference_epoch = 0
    loss_fn = partial(adjoint_loss, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                      tor_weight=args.tor_weight, sc_tor_weight=args.sc_tor_weight,
                      flexible_sidechains=args.flexible_sidechains)

    print("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0: print("Run name: ", args.run_name)
        logs = {}
        train_losses = finetune_epoch(model_base, model_finetune, train_loader, energy_fn, optimizer, device, t_to_sigma, loss_fn,
                                      ema_weights, args)
        print("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  sc_tor {:.4f}"
              .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                      train_losses['tor_loss'], train_losses['sc_tor_loss']))

        ema_weights.store(model_finetune.parameters())
        if args.use_ema: ema_weights.copy_to(
            model_finetune.parameters())  # load ema parameters into model for running validation and inference
        val_losses = test_epoch(model_finetune, val_loader, device, t_to_sigma, loss_fn, args.test_sigma_intervals)
        print("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   sc_tor {:.4f}"
              .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss'],
                      val_losses['sc_tor_loss']))

        if args.val_inference_freq is not None and (epoch + 1) % args.val_inference_freq == 0:
            inf_metrics = inference_epoch_fix(model_finetune, val_loader.dataset[:args.num_inference_complexes], device,
                                              t_to_sigma, args)
            print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']), end=" ")
            if args.flexible_sidechains:
                print(
                    "sc_rmsds_lt2 {:.3f} sc_rmsds_lt1 {:.3f}, sc_rmsds_lt0.5 {:.3f} avg_improve {:.3f} avg_worse {:.3f} "
                    .format(inf_metrics['sc_rmsds_lt2'],
                            inf_metrics['sc_rmsds_lt1'],
                            inf_metrics['sc_rmsds_lt05'],
                            inf_metrics['sc_rmsds_avg_improvement'],
                            inf_metrics['sc_rmsds_avg_worsening']),
                    end=" ")

                if args.compare_true_protein:
                    print(
                        "sc_rmsds_lt2_from_holo {:.3f} sc_rmsds_lt1_from_holo {:.3f}, sc_rmsds_lt05_from_holo.5 {:.3f} sc_rmsds_avg_improvement_from_holo {:.3f} sc_rmsds_avg_worsening_from_holo {:.3f} "
                        .format(inf_metrics['sc_rmsds_lt2_from_holo'],
                                inf_metrics['sc_rmsds_lt1_from_holo'],
                                inf_metrics['sc_rmsds_lt05_from_holo'],
                                inf_metrics['sc_rmsds_avg_improvement_from_holo'],
                                inf_metrics['sc_rmsds_avg_worsening_from_holo']),
                        end=" ")

            # Print newline
            print()

            logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

        if args.train_inference_freq is not None and (epoch + 1) % args.train_inference_freq == 0:
            if args.no_torsion:
                inf_metrics = inference_epoch_fix(model_finetune, train_loader.dataset[:args.num_inference_complexes],
                                                  device, t_to_sigma, args)
                print("Epoch {}: Train inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} "
                      .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']))
                logs.update({'traininf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)
            else:
                print(
                    'Skipping inference on the training dataset: not possible when running with torsion because the orig_pos is not saved for the training set.')

        if not args.use_ema: ema_weights.copy_to(model_finetune.parameters())
        ema_state_dict = copy.deepcopy(
            model_finetune.module.state_dict() if device.type == 'cuda' else model_finetune.state_dict())
        ema_weights.restore(model_finetune.parameters())

        if args.wandb:
            import wandb
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            wandb.log(logs, step=epoch + 1)

        state_dict = model_finetune.module.state_dict() if device.type == 'cuda' else model_finetune.state_dict()
        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[
                    args.inference_earlystop_metric] <= best_val_inference_value or
                 args.inference_earlystop_goal == 'max' and logs[
                     args.inference_earlystop_metric] >= best_val_inference_value):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))
        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))
        if args.flexible_sidechains and val_losses['sc_tor_loss'] <= best_val_sc_loss:
            print("Storing best sc_tor_loss model")
            best_val_sc_loss = val_losses['sc_tor_loss']
            torch.save(state_dict, os.path.join(run_dir, 'best_model_sc.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model_sc.pt'))
        if 'valinf_sc_rmsds_lt05_from_holo' in logs and logs[
            'valinf_sc_rmsds_lt05_from_holo'] >= best_val_inference_sc_value:
            print("Storing best sc_rmsds_lt05_from_holo model")
            best_val_inference_sc_value = logs['valinf_sc_rmsds_lt05_from_holo']
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model_sc.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model_sc.pt'))
        if 'valinf_rec_sc_lig_steric_clashes' in logs and logs[
            'valinf_rec_sc_lig_steric_clashes'] <= best_val_inference_steric_clashes_value:
            print("Storing best steric clashes model")
            best_val_inference_steric_clashes_value = logs['valinf_rec_sc_lig_steric_clashes']
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model_steric_clashes.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model_steric_clashes.pt'))

        if scheduler:
            if args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses['loss'])

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))


def main_function():
    args = parse_train_args()
    device = get_default_device()
    args.device = device
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (
                    args.scheduler_patience > args.val_inference_freq)  # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)

    energy_fn = get_energy_function(args.energy_fn, args)

    with open(f'{args.base_model_dir}/model_parameters.yml') as f:
        base_model_args = Namespace(**yaml.full_load(f))

    model_base = get_model(base_model_args, device, t_to_sigma=t_to_sigma)
    model_finetune = get_model(base_model_args, device, t_to_sigma=t_to_sigma)

    state_dict = torch.load(f'{args.base_model_dir}/{args.base_model_ckpt}', map_location=device)
    model_base.load_state_dict(state_dict, strict=True)
    model_finetune.load_state_dict(state_dict, strict=True)

    model_base = model_base.to(device)
    model_finetune = model_finetune.to(device)

    model_base.eval()
    for p in model_base.parameters():
        p.requires_grad_(False)

    optimizer, scheduler = get_optimizer_and_scheduler(args, model_finetune,
                                                       scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
    ema_weights = ExponentialMovingAverage(model_finetune.parameters(), decay=args.ema_rate)

    if args.restart_dir and os.path.exists(args.restart_dir):
        try:
            dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=device)
            if args.restart_lr is not None: dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(dict['optimizer'])
            model_finetune.module.load_state_dict(dict['model'], strict=True)
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(dict['ema_weights'], device=device)
            print("Restarting from epoch", dict['epoch'])
        except Exception as e:
            print("Exception", e)
            dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=device)
            model_finetune.module.load_state_dict(dict, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")

    numel = sum([p.numel() for p in model_finetune.parameters()])
    print(f'Model {type(model_finetune)} with {numel:,} parameters')

    if args.wandb:
        import wandb
        run = wandb.init(
            entity='bd825-imperial-college-london',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            tags=['finetune'],
            config=args
        )
        wandb.log({'numel': numel})

        run.alert(
            title="Run Started",
            text=f"Run {run.name} has started on {run.dir}",
            level=wandb.AlertLevel.INFO
        )

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)

    finetune(args, model_base, model_finetune, energy_fn, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma,
             run_dir)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main_function()
