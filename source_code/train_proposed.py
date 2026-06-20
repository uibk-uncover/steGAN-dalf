"""Training script for the GAN.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Tuple

import _data
import _models
import _stego
import _tools


def format_log_line(res, loss_a, loss_e):
    """"""
    # calculate the scores
    beta2_r, beta2_g, beta2_b = torch.mean(res.beta2_ch, dim=0)  # 2-ary change rate per channel
    alpha3_r, alpha3_g, alpha3_b = torch.mean(res.alpha3_ch, dim=0)  # 3-ary embedding rate per channel
    # format the line
    d = {
        **({'loss_a': loss_a.item()} if loss_a else {}),
        'loss_e': loss_e.item(),
        'α27': torch.mean(res.alpha27).item(),
        'α3': torch.mean(res.alpha3).item(),
        'α3r': torch.mean(alpha3_r).item(),
        'α3g': torch.mean(alpha3_g).item(),
        'α3b': torch.mean(alpha3_b).item(),
        'β27': torch.mean(1 - res.p27[:, 0]).item(),
        'β2': torch.mean(res.beta2).item(),
        'β2r': beta2_r.item(),
        'β2g': beta2_g.item(),
        'β2b': beta2_b.item(),
    }
    return d

def format_gradients(res: SimpleNamespace) -> str:
    """"""
    gradient_l1 = lambda x, dim=(0, 1): x.grad.abs().sum(dim=(2, 3)).mean(dim=dim)
    g_x0r, g_x0g, g_x0b = gradient_l1(res.x0, dim=0)
    g = {
        'p27': gradient_l1(res.p27),
        'λ': gradient_l1(res.lbda),
        'ρ27': gradient_l1(res.rho27),
        'x0': gradient_l1(res.x0),
        'x0r': g_x0r,
        'x0g': g_x0g,
        'x0b': g_x0b,
    }
    return g


def hinge1(l0: torch.Tensor, l1: torch.Tensor) -> Tuple[torch.Tensor]:
    """logits of x0 and x1 of shape [B, 2]

    :param l0:
    :param l1
    """
    score0, score1 = l0[:, 0], l1[:, 0]
    loss_a = torch.mean(score1)  # maximize fakes
    loss_e = torch.mean(F.relu(1. + score0)) + torch.mean(F.relu(1. - score1))
    return loss_a, loss_e


def beta_budget(beta2: torch.Tensor, budget: float = 0) -> torch.Tensor:
    """beta2 of shape [B, C, H, W]

    :param beta2:
    :param budget:
    """
    return F.relu(torch.mean(beta2) - budget)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize the probability map of the proposed method."
    )

    parser.add_argument('--cover', required=True, type=Path, help='')
    parser.add_argument('--model_dir', default=Path('../models'), type=Path)
    parser.add_argument('--device', default='cpu', type=str)

    return parser.parse_args()


def main():
    rng, g = _tools.seed_everything(12345)  # seed

    # parameters
    dry_run = False
    print_freq = 10
    save_freq = 250
    take_num_images = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _stego.initialize(device)  # precompute persistent projection matrices
    args = {
        'data_dir': Path('../data'),
        'model_dir': Path('../models'),
        'filters_cover': {'height': 296, 'width': 296},
        'num_epochs': 15,
        'batch_size': 32,
        'budget': .5,  # change rate budget (avg. # changes per triplet)
        'num_workers': 0,  #
        'shuffle_seed': 12345,
        'alpha_range': {'low': .1, 'high': 1.},
        'net_a': {
            'in_channels': 3,
            'out_channels': 27,
        },
        'net_e': {
            'in_channels': 3,
            'out_channels': 1,
        },
        'optimizer_a': {'lr': 1e-4},
        'optimizer_e': {'lr': 1e-4},
    }

    # model
    run_name = f'adversarial-{datetime.now().strftime("%y%m%d%H%M%S")}'
    run_dir = args['model_dir'] / run_name
    run_model_dir = run_dir / 'model'
    args_file = run_dir / 'config.json'
    if not dry_run:
        run_dir.mkdir(exist_ok=False)
        run_model_dir.mkdir(exist_ok=False)
        with open(args_file, 'w') as f:
            json.dump(args, f, indent=2, sort_keys=True, default=str)
    if not dry_run:
        log_f = open(run_dir / 'stdout.log', 'a')
    #
    log = _tools.StdoutLogger(module=Path(__file__).name)
    log.info(f'Training of {run_name} started.')

    # data
    tr_loader, tr_dataset = _data.get_data_loader('split_tr.csv', {**args}, drop_last=True)
    va_loader, va_dataset = _data.get_data_loader('split_va.csv', {**args, 'alpha': .4}, take_num_images=128, drop_last=True)

    # networks
    net_a = _models.DepthwiseSeparableBlock(**args['net_a']).to(device)
    net_e = _models.UcNetD(**args['net_e']).to(device)
    gan = _stego.StegoGAN27(net_a=net_a, net_e=net_e)

    # optimizers
    optimizer_a = torch.optim.RMSprop(net_a.parameters(), **args['optimizer_a'])
    optimizer_e = torch.optim.RMSprop(net_e.parameters(), **args['optimizer_e'])

    #
    best_pe_va = -np.inf
    epoch_start = 0
    step_start = 0
    for epoch in range(epoch_start, args['num_epochs']):
        tr_dataset.reshuffle()
        log.info(f'[{epoch:d}/{args["num_epochs"]:02d}] {args["alpha_range"]} Adversary training started.')

        # adversary training over minibatches
        loss_a_meter = _tools.LossMeter(gamma=.9, device=device)
        loss_e_meter = _tools.LossMeter(gamma=.9, device=device)
        pe_meter = _tools.PEMeter(history=1000, device=device)
        auc_meter = _tools.AUCMeter(history=1000, device=device)
        for i, (x0, alphas, names) in enumerate(tr_loader):
            step = epoch * len(tr_loader) + i
            #
            B, C, H, W = x0.size()
            x0 = x0.to(device)
            alphas = alphas[:, 0].to(device)

            #  === train Alice ===
            if step % 5 == 0:
                optimizer_a.zero_grad(set_to_none=True)
                net_a.train()
                net_e.eval()
                res = gan(
                    x0,
                    alpha=alphas,
                    simulator_method='gumbel-softmax',
                    stego_seed=12345 + step,
                    mode='a',
                )
                loss_a, _ = hinge1(res.l0, res.l1)
                if args['budget'] is not None:
                    loss_a += beta_budget(res.beta2, budget=args['budget'])
                #
                loss_a.backward()
                scores_g = format_gradients(res)
                torch.nn.utils.clip_grad_norm_(net_a.parameters(), max_norm=1.0)
                optimizer_a.step()

            # === train Eve ===
            optimizer_e.zero_grad(set_to_none=True)
            net_a.eval()
            net_e.train()
            res = gan(
                x0,
                alpha=alphas,
                simulator_method='gumbel-softmax',
                stego_seed=12345 + step,
                mode='e',
                no_eve=False,
            )
            #
            _, loss_e = hinge1(res.l0, res.l1)
            #
            loss_e.backward()
            torch.nn.utils.clip_grad_norm_(net_e.parameters(), max_norm=1.0)
            optimizer_e.step()

            # log statistics
            with torch.no_grad():
                #
                y = torch.cat([torch.zeros(B), torch.ones(B)], dim=0).to(torch.long).to(device)
                scores = torch.cat([res.l0[:, 0], res.l1[:, 0]], dim=0)
                loss_a_meter.update(loss_a)
                loss_e_meter.update(loss_e)
                pe_meter.update(y, scores)
                auc_meter.update(y, scores)
                scores = format_log_line(res, loss_a, loss_e)
                scores = {
                    **scores,
                    pe_meter.name: pe_meter.avg,
                    auc_meter.name: auc_meter.avg,
                    'sd/loss_a': torch.sqrt(loss_a_meter.var).item(),
                    'sd/loss_e': torch.sqrt(loss_e_meter.var).item(),
                    'g/ρ27': scores_g['ρ27'].item() if scores_g else None,
                }

            # check model
            if step > 0 and step % print_freq == 0:
                log.log_scalars(
                    scores,
                    split='tr',
                    step=step,
                    epoch=epoch,
                    fmt='%.04f',
                    num_batches=len(tr_loader),
                )
            if step > 0 and step % save_freq == 0:
                if not dry_run:
                    log_f.flush()
                    latest_model_file = run_model_dir / f'model_step_{step}.pt.tar'
                    torch.save({
                        'epoch': epoch,
                        'batch_index': i,
                        'step': step,
                        'net_a': {
                            'state_dict': net_a.state_dict(),
                            'optimizer': optimizer_a.state_dict(),
                        },
                        'net_e': {
                            'state_dict': net_e.state_dict(),
                            'optimizer': optimizer_e.state_dict(),
                        },
                        'dataset': {
                            'state_dict': tr_dataset.state_dict(),
                        },
                        'torch_rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    }, latest_model_file)
    #
    if not dry_run:
        log_f.close()


if __name__ == '__main__':
    main()

