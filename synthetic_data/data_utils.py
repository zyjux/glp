"""Utilities for handling synthetic data"""

""" Adapted from www.github.com/thunderhoser/ml4tc by Ryan Lagerquist """

from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from torch.utils.data import Dataset

DATA_FILE = "/mnt/data2/lverhoef/synthetic_ellipses/train_valid_ds.nc"


class Ellipse_Dataset(Dataset):
    def __init__(
        self,
        full_ds: xr.Dataset,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        transforms=None,
    ):
        super().__init__()
        self.full_ds = full_ds.isel(sample=slice(start_idx, end_idx))
        self.transforms = transforms

    def __len__(self):
        return self.full_ds.sample.shape[0]

    def __getitem__(self, idx: int):
        image = torch.unsqueeze(
            torch.tensor(
                self.full_ds.ellipse.isel(sample=idx).values.astype(np.float32)
            ),
            0,
        )
        label = torch.tensor(
            self.full_ds.target.isel(sample=idx).values, dtype=torch.long
        )
        if self.transforms is not None:
            images = torch.stack(
                [image] + [transform(image) for transform in self.transforms]
            )
            labels = torch.stack([label] * (1 + len(self.transforms)))
        else:
            images = torch.stack([image])
            labels = torch.stack([label])
        return images, labels


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.normal(self.mean, self.std, size=tensor.shape)


def angle_categorizer(
    angle: npt.NDArray, num_categories: int, diagonal: bool = False
) -> npt.NDArray:
    """Helper function to categorize array of angles defining major axes of ellipses

    This function assumes that angles define a line through the origin, and as such each
    line corresponds to a unique angle in the 0 to pi range. Angles outside that range
    will be treated as equivalent to the angle in [0, pi) that defines the same line.

    args:
        angle (numpy.typing.NDArray): Array of angles to categorize.
        num_categories (int): How many subdivisions of the 0 to pi range angles should
            be broken into.
        diagonal (bool): Whether the first subdivision should start at 0 radians (False)
            or be centered on 0 (True). Default is False.

    returns:
        numpy.typing.NDArray: Array of integers indicating which region the angle falls
            into, starting at 0 and increasing counter-clockwise up to one less than
            num_categories.
    """
    region_angle = np.pi / num_categories
    comp_angles = angle / region_angle
    if diagonal:
        comp_angles += 1 / 2
    comp_angles = comp_angles.astype(int) % num_categories
    return comp_angles
