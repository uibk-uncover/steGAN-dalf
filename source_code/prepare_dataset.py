"""Generator of the dataset.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
import torch
from tqdm import tqdm

import _data
from embed import simulate


def prepare_covers(data_dir: Path):
    """Generator of covers.

    :param data_dir: Dataset directory.
    """

    # initialize
    input_url = f'http://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_512_COLOR/'
    cover_dir = data_dir / 'images'

    #
    df = pd.concat([
        pd.read_csv('../data/split_tr.csv'),
        pd.read_csv('../data/split_va.csv'),
        pd.read_csv('../data/split_te.csv'),
    ]).sort_values('name').reset_index(drop=True)

    #
    try:
        files = []
        cover_dir.mkdir(exist_ok=False)
        for _, fname in enumerate(tqdm(df['name'])):
            #
            res = requests.get(f"{input_url}/{Path(fname).stem}.tif", verify=False)  # the ALASKA dataset website has an expired certificate (I already informed them)
            assert res.ok
            # decode TIFF
            stream = io.BytesIO()
            stream.write(res.content)
            x0 = np.array(Image.open(stream, formats=['TIFF']))
            # top-left crop to 296x296
            x0 = x0[:296, :296]
            Image.fromarray(x0).save(cover_dir / fname)
            #
            files.append({
                'name': str((cover_dir / fname).relative_to(data_dir)),
                'height': x0.shape[0],
                'width': x0.shape[1],
            })
    except KeyboardInterrupt:
        pass
    finally:
        if files:
            files = pd.DataFrame(files)
            files.to_csv(cover_dir / 'files.csv', index=False)


def prepare_stego(data_dir: Path, cover_dir: Path, alpha: float, device: torch.device):
    """Generator of stego images.

    :param data_dir: Dataset directory (for the output).
    :param cover_dir: Cover image directory.
    :param alpha: Embedding rate.
    :param device: Target device.
    """
    #
    with open('../models/steGANdalf/config.json') as f:
        args = json.load(f)
    args = {
        **args,
        'filters_cover': {'height': 296, 'width': 296},
        'crop': None,
        'data_dir': data_dir,
        'alpha': alpha,
        'batch_size': 4,
        'num_workers': 0,
        'shuffle_seed': 12345,
    }

    # generate dataset
    all_loader, all_dataset = _data.get_data_loader((cover_dir / 'files.csv').relative_to(data_dir), args)
    stego_method = 'steGANdalf_step_24000'
    stego_dir = Path(f'../data/stego_{stego_method}_{alpha}_joint27_images')
    stego_dir.mkdir(exist_ok=False)
    try:
        files = []
        print(f'[24000] {alpha=:.03f} Generating dataset of fixed U-Net, directory {stego_dir}.')
        for i, (x0, alphas, names) in tqdm(enumerate(all_loader), total=len(all_loader)):
            B, C, H, W = x0.size()
            N = C * H * W
            x0 = x0.to(device)
            alphas = alphas[:, 0].to(device)

            # === apply Alice ===
            x1_ = simulate(
                x0=x0,
                alpha=alphas,
                stego_seed=12345 + i,
                device=device,
            )
            #
            x1 = x1_.cpu().numpy().transpose(0, 2, 3, 1)
            for b in range(B):
                image_name = Path(names[b]).name
                x1_b = x1[b]
                Image.fromarray(x1_b).save(stego_dir / image_name)
                files.append({
                    'name': str((stego_dir / image_name).relative_to(data_dir)),
                    'height': x1_b.shape[0],
                    'width': x1_b.shape[1],
                    'stego_method': stego_method,
                    'color_strategy': 'joint27',
                    'simulator': 'mi',
                    'alpha': alpha,
                })
    finally:
        if files:
            files = pd.DataFrame(files)
            files.to_csv(stego_dir / 'files.csv', index=False)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Script to generate the ALASKA dataset."
    )

    parser.add_argument('--data_dir', default='../data', type=Path, help='dataset directory')
    parser.add_argument('--alpha', default=.4, type=float, help='embedding rate')
    parser.add_argument('--device', default='cpu', type=str, help='target device')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    device = torch.device(args.device)

    prepare_covers(data_dir=args.data_dir)
    prepare_stego(
        data_dir=args.data_dir,
        cover_dir=args.data_dir / 'images',
        alpha=args.alpha,
        device=device,
    )


if __name__ == '__main__':
    main()
