"""Data-related functions.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Dict, Any, Callable, Optional, Tuple
import warnings

pd.options.mode.chained_assignment = None  # suppress Pandas warning


class CoverDataset(torch.utils.data.Dataset):
    """Cover dataset."""
    def __init__(
        self,
        data_dir: Path | str,
        split: str,
        cover_dir: str = None,
        shuffle_seed: int = None,
        transform: Optional[Callable] = None,
    ):
        """Construction.

        :param data_dir: dataset directory
        :param split: split path within the dataset
        :param shuffle_seed:
        :param transform:
        """
        # parameters
        self.data_dir = Path(data_dir)
        self.transform = transform

        # get selected dataset
        self.config_cover = pd.read_csv(self.data_dir / split, low_memory=False)
        if cover_dir is not None:
            cover_dir = Path(cover_dir)
            self.config_cover['name'] = self.config_cover['name'].apply(
                lambda name: str(cover_dir / name))
        assert len(self.config_cover) > 0, 'no such covers found'

        to_remove = []
        for i, row in self.config_cover.iterrows():
            if not (self.data_dir / row['name']).exists():
                warnings.warn(f'{row['name']} not found, ignoring')
                to_remove.append(i)
        self.config_cover = self.config_cover.drop(to_remove)

        # for dataset reshuffling
        self._rng = np.random.default_rng(shuffle_seed)
        self.reshuffle()

        # check dataset
        assert len(self.config_cover) > 0, 'no such covers found, did you forget running prepare_dataset.py?'

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.config_cover)

    def state_dict(self):
        """Get dataset state."""
        return {'rng': self._rng_prev_state}
    def load_state_dict(self, state):
        """Load dataset from state.

        :param state: dataset state
        """
        self._rng.bit_generator.state = state['rng']
    def reshuffle(self):
        """Call after each epoch to reshuffle."""
        self._rng_prev_state = self._rng.bit_generator.state  # last rng state
        # shuffle cover
        shuffle_seed = self._rng.integers(2**32-1)
        self.config_cover = self.config_cover.sort_values('name')
        self._reset_indices(drop=True)
        self.config_cover = self.config_cover.sample(
            frac=1,
            random_state=shuffle_seed,
        )
        self._reset_indices(drop=True)

    def cover_index(self):
        """Construct index of the covers."""
        df = self.config_cover.apply(lambda r: Path(r["name"]).stem, axis=1)
        return df

    def _reset_indices(self, drop=True):
        """Reset indices of cover/stego datasets."""
        self.config_cover = self.config_cover.reset_index(drop=drop)

    def __getitem__(self, idx: int):
        """Access the image at the given index.

        :param idx: Index.
        """
        # find cover
        image_row = self.config_cover.iloc[idx, :]
        image_name = image_row['name']

        # load image
        image = np.array(Image.open(self.data_dir / image_name))

        # transform
        if self.transform is not None:
            image = self.transform(image)

        # retreive image+target pairs
        return (*image, image_name)


class UniformAlpha(torch.nn.Module):
    """Data augmentation for bootstrapping over alpha."""
    def __init__(
        self,
        value: float = None,
        low: float = 0,
        high: float = 1,
    ):
        """Constructor.

        :param value: Fixed alpha (if given).
        :param low: Low bound for bootstrapping.
        :param high: High bound for bootsrapping.
        """
        super().__init__()
        self.value = value
        self.low, self.high = low, high

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """"""
        alpha = torch.ones((1,), device=x.device)
        alpha = (alpha * self.value) if self.value else alpha.uniform_(self.low, self.high)
        return x, alpha

    def __repr__(self):
        return f'FixedAlpha({self.value})' if self.value else f'UniformAlpha({self.low}, {self.high})'


def get_transform(
    *,
    alpha: float = None,
    alpha_range: float = {},
):
    """Construct data transform.

    :param alpha: Fixed embedding rate
    :param alpha_range: Embedding rate bounds, low and high, for bootstrapping.
    """
    # dataset transforms
    transform = [transforms.ToTensor()]
    transform += [UniformAlpha(value=alpha, **alpha_range)]
    return transforms.Compose(transform)


def get_data_loader(
    split: str,
    config: Dict[str, Any],
    cover_dir: str = None,
    **kw,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset]:
    """Construct dataset and dataloader.

    :param split: split within the set
    :param config: dataset configuration
    """
    # Dataset transform
    transform = [transforms.ToTensor(),]
    alpha_range = config.get('alpha_range', {})
    transform += [UniformAlpha(value=config.get('alpha', None), **alpha_range)]
    transform = transforms.Compose(transform)

    print('Data transform')
    print(transform.transforms)

    # Dataset
    dataset = CoverDataset(
        # dataset
        data_dir=config['data_dir'],
        split=split,
        cover_dir=cover_dir,
        # pair constraint
        shuffle_seed=config.get('shuffle_seed', None),
        # other
        transform=transform,
    )

    # Create data loaders
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True,
        drop_last=False,
        **kw,
    )

    #
    return loader, dataset
