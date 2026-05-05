import logging

from tqdm import tqdm

from utils import so3, torus
from utils.training import AverageMeter
import torch
from utils.diffusion_utils import get_t_schedule, get_inverse_schedule


def finetune_epoch(model_base, model_finetune, loader, optimizer, device, t_to_sigma, loss_fn, ema_weigths, args):
    model_finetune.train()
    model_base.eval()
    meter = AverageMeter(['total_loss', 'total_base_loss', 'tr_loss', 'rot_loss', 'tor_loss', 'sc_tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss','sc_tor_base_loss'])

    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    if args.asyncronous_noise_schedule:
        tr_schedule = get_inverse_schedule(t_schedule, args.sampling_alpha, args.sampling_beta)
        rot_schedule = get_inverse_schedule(t_schedule, args.rot_alpha, args.rot_beta)
        tor_schedule = get_inverse_schedule(t_schedule, args.tor_alpha, args.tor_beta)
        sidechain_tor_schedule = get_inverse_schedule(t_schedule, args.sidechain_tor_alpha, args.sidechain_tor_beta)
    else:
        tr_schedule, rot_schedule, tor_schedule, sidechain_tor_schedule = t_schedule, t_schedule, t_schedule, t_schedule

    for data_row in tqdm(loader, total=len(loader)):
        data = data_row
        logging.debug(f"Batch size: {len(data)}. Batch data type: {type(data)}")
        # On CPU data is a batch of graphs, on GPU it is a list of graphs (?)
        if (device.type == 'cuda' and len(data) == 1) or (device.type == 'cpu' and data.num_graphs == 1):
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            loss, log_dict = loss_fn(data_row, model_base, model_finetune, args.inference_steps, tr_schedule, rot_schedule, tor_schedule, sidechain_tor_schedule, device, t_to_sigma, args)

            if loss.isnan():
                print("SKIPPING backward pass for batch, loss is nan. This could indicate that the batch has no ligand torsion or sidechain torsions")
            else:
                if loss.isinf():
                    print("WARN: Loss is infinite.")

                loss.backward()
                optimizer.step()
                ema_weigths.update(model_finetune.parameters())
                meter.add([log_dict[typ] for typ in meter.types])
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print('| WARNING: ran out of memory, skipping batch')
                for p in model_finetune.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'input mismatch' in str(e).lower():
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model_finetune.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return meter.summary()
