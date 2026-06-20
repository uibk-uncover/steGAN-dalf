"""Visualize the probability maps.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
from glob import glob
import numpy as np
from pathlib import Path
from PIL import Image
import torch

import embed_proposed


def probability(
    x0: np.ndarray,
    alpha: float,
    device: torch.device = None,
) -> np.ndarray:
    """

    :param x0:
    :param alpha:
    :param device:
    """
    # initialize
    if embed_proposed.GAN is None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embed_proposed.initialize_model(device=device)
    # convert to torch
    x0_ = torch.from_numpy(x0.astype('float32').transpose(2, 0, 1) / 255.)[None].to(device)
    # calculate the probabilities
    res_a = embed_proposed.GAN.get_probabilities(
        x0=x0_,
        alpha=alpha,
        mode='pass',
    )
    # convert to numpy
    p2 = res_a.p2[0].numpy()  # 3 H W
    """p2 are the binary (projected) probabilities"""
    assert p2.shape == (3, 296, 296)
    return p2


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize the probability map of the proposed method."
    )

    parser.add_argument('--cover_dir', default=Path('../example_images/cover'), type=Path, help='TODO')
    parser.add_argument('--cover_image', default=None, type=Path, help='TODO')
    parser.add_argument('--out_dir', default=Path('../example_images/probability'), type=Path, help='TODO')
    parser.add_argument('--alpha', default=.4, type=float, help='TODO')
    parser.add_argument('--clip', default=.3, type=float, help='TODO')
    parser.add_argument('--device', default='cpu', type=str, help='TODO')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(exist_ok=True)

    cover_list = [args.cover_image] if args.cover_image else map(Path, glob(str(args.cover_dir / '*.png')))
    for cover_name in cover_list:

        x0 = np.array(Image.open(cover_name))
        p2 = probability(x0, alpha=.4, device=device)

        p2 = np.clip(np.round(p2.transpose(1, 2, 0) / args.clip * 255), 0, 255).astype('uint8')
        Image.fromarray(p2).save(args.out_dir / f'{cover_name.stem}.png')


if __name__ == '__main__':
    main()
