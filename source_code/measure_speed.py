"""Script for benchmarking the embedding speed.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
import cv2
import json
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path
import time
import torch

import _stego
import _models


#
class AlaskaDataset(torch.utils.data.Dataset):
    """ALASKA dataset from a data frame."""
    def __init__(self, file_list, resolution: int):
        """Constructor.

        :param file_list: List of files.
        :param resolution: Target resolution.
        """
        self.file_list = file_list
        self.resolution = resolution
    def __len__(self):
        """Total size of the dataset."""
        return len(self.file_list)
    def __getitem__(self, idx):
        """Access the image at the given index. Pads the image to the targeted size.

        :param idx: Index.
        """
        img_path = self.file_list[idx]
        #
        image = cv2.imread(img_path, -1)
        H, W = image.shape[:2]
        padh, padw = self.resolution-H, self.resolution-W
        image = np.pad(image, ((padh//2, padh//2), (padw//2, padw//2), (0,0)), mode='symmetric')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image.astype(np.float32) / 255.)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Measure embedding speed of the proposed method."
    )

    parser.add_argument('--cover_dir', default=Path('../example_images/cover'), type=Path, help='cover directory')
    parser.add_argument('--resolution', default=512, type=int, help='cover image resolution')
    parser.add_argument('--device', default='cpu', type=str, help='target device')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    device = torch.device(args.device)
    _stego.initialize(device)

    # setup the model
    model_path = Path('../models/proposed')
    config = json.load(open(model_path / 'config.json'))
    net_a = _models.DepthwiseSeparableBlock(**config['net_a']).to(device)
    checkpoint = torch.load(model_path / 'model' / 'model_step_24000.pt.tar', map_location=torch.device('cpu'), weights_only=True)
    net_a.load_state_dict(checkpoint['net_a']['state_dict'])
    net_a.eval()

    # setup the dataset
    files = glob(str(args.cover_dir / '*.png'))
    dataset = AlaskaDataset(files, resolution=args.resolution)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        prefetch_factor=None,
        num_workers=0,
    )

    with torch.no_grad():
        execution_times = []
        for _, x0 in enumerate(dataloader):
            x0 = x0.to(device)
            B, C, H, W = x0.size()
            N = C * H * W

            # calculate cost
            start_cost = time.perf_counter()
            rho27 = net_a(x0)
            rho27 = _stego.adjust_rho27(rho27, x0)
            end_cost = time.perf_counter()

            # simulate
            _stego.find_lambda.solver = 'dde'
            start_sim = time.perf_counter()
            # objective = _stego.average_payload_27to3  # match asymmetric entropy
            objective = _stego.average_payload  # match native entropy
            lbda = _stego.find_lambda.apply(rho27, .4 * N, N, objective)
            p27, _ = objective(rho27, lbda)
            delta3, delta27 = _stego.simulate(
                p=p27,
                simulator_method='hard',
                stego_seed=12345,
            )
            x1 = x0 + delta3.to(x0.dtype) / 255.
            end_sim = time.perf_counter()

            # record the speed
            for _ in range(B):
                execution_times.append({
                    'cost': end_cost - start_cost,
                    'sim': end_sim - start_sim,
                })
    #
    execution_times = pd.DataFrame(execution_times)
    print(execution_times)


if __name__ == '__main__':
    main()

