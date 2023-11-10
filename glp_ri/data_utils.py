""" Utilities for handling data """

""" Adapted from www.github.com/thunderhoser/ml4tc by Ryan Lagerquist """

import os
import glob
import json
import xarray as xr
import torch
import numpy as np
from torch.utils.data import Dataset

CYCLONE_ID_REGEX = '[0-9][0-9][0-9][0-9][A-Z][A-Z][0-9][0-9]'
VALID_BASIN_ID_STRINGS = ['AL', 'SL', 'EP', 'CP', 'WP', 'IO', 'SH']
DATA_DIR = '/nfs/home/lverho/research_data/RI'


def find_file(directory_name, cyclone_id_string, raise_error_if_missing=True):
    """Finds NetCDF file with learning examples.

    :param directory_name: Name of directory with example files. Must be absolute path
    :param cyclone_id_string: Cyclone ID (must be accepted by
        `parse_cyclone_id`).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: example_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    assert type(directory_name) is str, f'directory_name must be str; got {type(directory_name)}'
    parse_cyclone_id(cyclone_id_string)
    assert type(raise_error_if_missing) is bool, f'raise_error_if_missing must be bool; got {type(raise_error_if_missing)}'

    example_file_name = f'{directory_name}/learning_examples_{cyclone_id_string}.nc'

    if os.path.isfile(example_file_name) or not raise_error_if_missing:
        return example_file_name

    error_string = f'Cannot find file.  Expected at: "{example_file_name}"'
    raise ValueError(error_string)


def parse_cyclone_id(cyclone_id_string):
    """Parses metadata from cyclone ID.

    :param cyclone_id_string: Cyclone ID, formatted like "yyyybbcc", where yyyy
        is the year; bb is the basin ID; and cc is the cyclone number ([cc]th
        cyclone of the season in the given basin).
    :return: year: Year (integer).
    :return: basin_id_string: Basin ID.
    :return: cyclone_number: Cyclone number (integer).
    """

    assert type(cyclone_id_string) is str, f'cyclone_id_string must be type str; got {type(cyclone_id_string)}'
    assert len(cyclone_id_string) == 8, f'cyclone_id_string must be length 8; got {len(cyclone_id_string)}'

    year = int(cyclone_id_string[:4])
    assert year >= 0, f'Year must be >= 0; got {year}'

    basin_id_string = cyclone_id_string[4:6]
    assert basin_id_string in VALID_BASIN_ID_STRINGS, f'Basin id must be one of {VALID_BASIN_ID_STRINGS}; got {basin_id_string}'

    cyclone_number = int(cyclone_id_string[6:])
    assert cyclone_number > 0, f'Cyclone number must be > 0; got {cyclone_number}'

    return year, basin_id_string, cyclone_number


def file_name_to_cyclone_id(example_file_name):
    """Parses cyclone ID from name of file with learning examples.

    :param example_file_name: File path.
    :return: cyclone_id_string: Cyclone ID.
    """

    assert type(example_file_name) is str, f'example_file_name must be str; got {type(example_file_name)}'
    pathless_file_name = os.path.split(example_file_name)[1]

    cyclone_id_string = pathless_file_name.split('.')[0].split('_')[-1]
    parse_cyclone_id(cyclone_id_string)

    return cyclone_id_string


def find_cyclones(directory_name, raise_error_if_all_missing=True):
    """Finds all cyclones.

    :param directory_name: Name of directory with example files.
    :param raise_error_if_all_missing: Boolean flag.  If no cyclones are found
        and `raise_error_if_all_missing == True`, will throw error.  If no
        cyclones are found and `raise_error_if_all_missing == False`, will
        return empty list.
    :return: cyclone_id_strings: List of cyclone IDs.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    assert type(directory_name) is str, f'directory_name must be str; got {type(directory_name)}'
    assert type(raise_error_if_all_missing) is bool, f'raise_error_if_all_missing must be bool; got {type(raise_error_if_all_missing)}'

    file_pattern = f'{directory_name}/learning_examples_{CYCLONE_ID_REGEX}.nc'
    example_file_names = glob.glob(file_pattern)

    cyclone_id_strings = []

    for this_file_name in example_file_names:
        try:
            cyclone_id_strings.append(
                file_name_to_cyclone_id(this_file_name)
            )
        except:
            pass

    cyclone_id_strings = list(set(cyclone_id_strings))
    cyclone_id_strings.sort()

    if raise_error_if_all_missing and len(cyclone_id_strings) == 0:
        error_string = (
            'Could not find any cyclone IDs from files with pattern: "{0:s}"'
        ).format(file_pattern)

        raise ValueError(error_string)

    return cyclone_id_strings


def load_labels(fn):
    with open(fn, 'r') as f:
        raw_dict = json.load(f)

    labels = []
    for storm_id in raw_dict.keys():
        for timestamp in raw_dict[storm_id].keys():
            labels.append((storm_id, timestamp, raw_dict[storm_id][timestamp]))

    return labels


class RI_Dataset(Dataset):
    def __init__(self, labels, transform=None, target_transform=None):
        if type(labels) is list:
            self.labels = labels
        if type(labels) is str:
            self.labels = load_labels(labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        storm_id, timestamp, label = self.labels[idx]
        ds = xr.open_dataset(find_file(DATA_DIR, storm_id)).sel(satellite_valid_time_unix_sec=int(timestamp))
        image = torch.reshape(torch.tensor(ds.satellite_predictors_gridded.values.astype(np.float32)), (1, 380, 540))
        label = torch.tensor(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label