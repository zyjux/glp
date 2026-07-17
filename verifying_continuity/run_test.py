import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import yaml
from network_def import simple_model
from torchinfo import summary
from torchvision.transforms.functional import rotate

parser = argparse.ArgumentParser(prog="RunTests", description="Runs tests")
parser.add_argument("cfg", help="Config file", type=str)
parser.add_argument("exp_dir", help="Directory of experiment directory", type=str)
args = parser.parse_args()

TC_DATA_DIR = "/mnt/data2/lverhoef/RI/learning_examples/rotated_with_storm_motion/imputed/normalized/"


@dataclass
class configs:
    g_angle_inc: int
    num_lambda_cosets: int
    thetas: list[float]
    manual_seed: int
    input_source: Literal["random", "ellipse", "tc"]
    num_input_samples: int
    img_size: tuple[int, int]
    plot_fn: str = "difference_plot.png"
    do_pool: bool = True


def compare_rotated_outputs(outputs: torch.Tensor):
    ref_output = outputs[0, ...]
    rotated_output = outputs[1:, ...]
    diffs = torch.abs(rotated_output - ref_output)
    return torch.amax(diffs, dim=tuple(range(1, diffs.ndim)))


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

    assert (
        type(directory_name) is str
    ), f"directory_name must be str; got {type(directory_name)}"
    assert (
        type(raise_error_if_missing) is bool
    ), f"raise_error_if_missing must be bool; got {type(raise_error_if_missing)}"

    example_file_name = f"{directory_name}/learning_examples_{cyclone_id_string}.nc"

    if os.path.isfile(example_file_name) or not raise_error_if_missing:
        return example_file_name

    error_string = f'Cannot find file.  Expected at: "{example_file_name}"'
    raise ValueError(error_string)


def load_labels(
    fn: str | Path,
):
    """Data utility to parse labels json file into numpy array

    args:
        fn (str): Filename of the json dictionary to load.

    returns:
        labels (np.ndarray): A 3xN array, where N is the number of samples in the json
            file. The first column contains storm ids, the second column contains
            timestamps, and the third column contains the label.

    """
    with open(fn, "r") as f:
        raw_dict = json.load(f)

    labels = []
    for storm_id in raw_dict.keys():
        for timestamp in raw_dict[storm_id].keys():
            labels.append((storm_id, timestamp, raw_dict[storm_id][timestamp]))

    labels = np.array(labels, dtype=object)

    return labels


# Load experiment configuration file
with open(args.cfg, "r") as f:
    cfg = configs(**yaml.safe_load(f))

torch.manual_seed(cfg.manual_seed)
torch.cuda.manual_seed(cfg.manual_seed)

if cfg.input_source == "ellipse":
    indices = torch.randint(10000, (cfg.num_input_samples,))
elif cfg.input_source == "random":
    random_images = torch.randn((cfg.num_input_samples, 1, 1, *cfg.img_size))
elif cfg.input_source == "tc":
    labels_file = Path(TC_DATA_DIR, "valid_labels.json")
    labels = load_labels(labels_file)
    indices = torch.randint(labels.shape[0], (cfg.num_input_samples,))
    cfg.img_size = (380, 540)

# Detect gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

# Set up plot
F, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlabel("Theta")
ax.set_ylabel("Max difference")
# ax.set_ylim(None, 0.015)

# theta_max = max(cfg.thetas)
# ref_thetas = np.arange(0, theta_max, 0.1)
# alpha = 0.05
# rad_thetas = np.deg2rad(ref_thetas)
# ref_curve = alpha * np.sqrt(2 * (1 - np.cos(rad_thetas)))
# ax.plot(ref_thetas, ref_curve, "k--")

# Create model
model = simple_model(
    G_angle_inc=cfg.g_angle_inc,
    num_lambda_cosets=cfg.num_lambda_cosets,
    psi_size=cfg.img_size,
    do_pool=cfg.do_pool,
)

summary(model, input_size=(1, *cfg.img_size))

for sample in range(cfg.num_input_samples):
    # Generate input
    if cfg.input_source == "random":
        input_f = random_images[sample, ...].to(device)
        print(f"\n Processing random sample {sample}")
    elif cfg.input_source == "ellipse":
        input_f = torch.tensor(
            xr.open_dataset("/mnt/data2/lverhoef/synthetic_ellipses/train_valid_ds.nc")
            .isel(sample=indices[sample])["ellipse"]
            .to_numpy(),
            dtype=torch.float,
            device=device,
        ).view(1, 1, *cfg.img_size)
        print(f"\n Processing ellipse {indices[sample]}")
    elif cfg.input_source == "tc":
        storm_id, timestamp, _ = labels[indices[sample]]
        ds = xr.open_dataset(find_file(TC_DATA_DIR, storm_id)).sel(
            satellite_valid_time_unix_sec=int(timestamp)
        )
        input_f = torch.tensor(
            ds.satellite_predictors_gridded.values.astype(np.float32), device=device
        ).view(1, 1, *cfg.img_size)

    input_norm = input_f.norm(2)
    print(input_norm)

    # Create stacked rotations of input
    input_gf = torch.cat(
        [input_f] + [rotate(input_f, theta) for theta in cfg.thetas], dim=0
    )

    # Apply model
    with torch.no_grad():
        outputs = model(input_gf)

    print(outputs.shape)
    # Compare rotated elements against original
    max_diffs = compare_rotated_outputs(outputs) / input_norm
    # max_diffs /= (input_f**2).sum()

    # Print values
    print("\nTheta | Difference")
    print("------------------")
    for i, theta in enumerate(cfg.thetas):
        print(f"{theta: 6d} | {max_diffs[i].item():.7f}")

    ax.plot(cfg.thetas, max_diffs.cpu())

# Save completed figure
F.savefig(Path(args.exp_dir, cfg.plot_fn), bbox_inches="tight", dpi=300)
