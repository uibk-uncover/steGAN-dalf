"""Embeds into a cover image using the proposed method.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch

import _models
import _stego


GAN = None

def initialize_model(device: torch.device):
    """Initializes the model, global to this module.

    :param device: target device
    """
    global GAN
    #
    _stego.initialize(device)
    # load model
    model_path = Path('../models/proposed')
    args = json.load(open(model_path / 'config.json'))
    net_a = _models.DepthwiseSeparableBlock(**args['net_a']).to(device)
    checkpoint = torch.load(model_path / 'model' / 'model_step_24000.pt.tar', map_location=torch.device('cpu'), weights_only=True)
    net_a.load_state_dict(checkpoint['net_a']['state_dict'])
    net_a.eval()
    GAN = _stego.StegoGAN27(net_a=net_a, pad=16)


def simulate(
    x0: np.ndarray | torch.Tensor,
    alpha: float,
    stego_seed: float,
    device: torch.device,
) -> np.ndarray | torch.Tensor:
    """Simulate steganography with the proposed method.

    :param x0: cover image
    :param alpha: embedding rate
    :param stego_seed: steganographic seed
    :param device: target device
    """
    global GAN
    if GAN is None:
        initialize_model(device=device)
    #
    if isinstance(x0, np.ndarray):
        assert len(x0.shape) == 3
        assert x0.shape[2] == 3
        x0_ = torch.from_numpy(x0.astype('float32').transpose(2, 0, 1) / 255.)[None].to(device)
    elif isinstance(x0, torch.Tensor):
        assert len(x0.shape) == 4
        assert x0.shape[1] == 3
        x0_ = x0
    else:
        raise NotImplementedError
    # embed
    with torch.no_grad():
        res_a = GAN(
            x0=x0_,
            alpha=alpha,
            simulator_method='hard',
            stego_seed=stego_seed,
            no_eve=True,
            mode='pass',
        )
    x1 = torch.clamp(torch.round(res_a.x1 * 255.), 0, 255.).to(torch.uint8)
    if isinstance(x0, np.ndarray):
        x1 = x1.cpu().numpy()[0].transpose(1, 2, 0)
    return x1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize the probability map of the proposed method."
    )

    parser.add_argument('--cover', default=Path('../example_images/cover/17416_cover296.png'), type=Path, help='TODO')
    parser.add_argument('--stego', default=Path('../stego.png'), type=Path, help='TODO')
    parser.add_argument('--alpha', default=.4, type=float, help='TODO')
    parser.add_argument('--seed', default=12345, type=int, help='TODO')
    parser.add_argument('--device', default='cpu', type=str, help='TODO')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    device = torch.device(args.device)

    # load cover image
    x0 = np.array(Image.open(args.cover))
    x1 = simulate(
        x0=x0,
        alpha=args.alpha,
        stego_seed=args.seed,
        device=device,
    )
    print(f'Changed {(x0 != x1).mean()*100:.02f}% of elements.')

    # stego_path = f'../{args.cover.name}'  # TODO
    Image.fromarray(x1).save(args.stego)
    print(f'The stego object saved as "{args.stego}".')


if __name__ == '__main__':
    main()
