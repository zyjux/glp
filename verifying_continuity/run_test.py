import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
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


@dataclass
class configs:
    g_angle_inc: int
    num_lambda_cosets: int
    thetas: list[float]
    manual_seed: int
    input_source: Literal["random", "ellipse"]
    num_input_samples: int
    img_size: tuple[int, int]


def compare_rotated_outputs(outputs: torch.Tensor):
    ref_output = outputs[0, ...]
    rotated_output = outputs[1:, ...]
    diffs = torch.abs(rotated_output - ref_output)
    return torch.amax(diffs, dim=tuple(range(1, diffs.ndim)))


# Load experiment configuration file
with open(args.cfg, "r") as f:
    cfg = configs(**yaml.safe_load(f))

torch.manual_seed(cfg.manual_seed)
torch.cuda.manual_seed(cfg.manual_seed)

if cfg.input_source == "ellipse":
    indices = torch.randint(10000, (cfg.num_input_samples,))
elif cfg.input_source == "random":
    random_images = torch.randn((cfg.num_input_samples, 1, 1, *cfg.img_size))

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

# Create model
model = simple_model(
    G_angle_inc=cfg.g_angle_inc,
    num_lambda_cosets=cfg.num_lambda_cosets,
    psi_size=cfg.img_size,
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

    # Create stacked rotations of input
    input_gf = torch.cat(
        [input_f] + [rotate(input_f, theta) for theta in cfg.thetas], dim=0
    )

    # Apply model
    with torch.no_grad():
        outputs = model(input_gf)

    print(outputs.shape)
    # Compare rotated elements against original
    max_diffs = compare_rotated_outputs(outputs)
    # max_diffs /= (input_f**2).sum()

    # Print values
    print("\nTheta | Difference")
    print("------------------")
    for i, theta in enumerate(cfg.thetas):
        print(f"{theta: 6d} | {max_diffs[i].item():.7f}")

    ax.plot(cfg.thetas, max_diffs.cpu())

# Save completed figure
F.savefig(Path(args.exp_dir, "difference_curve.png"), bbox_inches="tight", dpi=300)
