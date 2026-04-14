import argparse
from pathlib import Path
from time import perf_counter

import torch
import xarray as xr
import yaml
from data_utils import DATA_FILE, Ellipse_Dataset, angle_categorizer
from network_def import CNN, crps_loss, model_config
from torch.utils.data import DataLoader
from torchinfo import summary

parser = argparse.ArgumentParser(prog="ApplyModel", description="Applies GLP models")
parser.add_argument("config_file", type=Path)
parser.add_argument("save_file", type=Path)
parser.add_argument("--data_file", default=Path(DATA_FILE), type=Path)
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

# Load config file
with open(args.config_file, "r") as f:
    cfg = model_config(**yaml.safe_load(f))

loss_function_translation = {"crps": crps_loss}

# Load dataset
ds = xr.load_dataset(DATA_FILE)

# Categorize angles
ds["target"] = (
    "sample",
    angle_categorizer(
        ds.angle.values,
        cfg.training_hyperparameters["num_classes"],
        diagonal=cfg.training_hyperparameters["diagonal_categorization"],
    ),
)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

pt_ds = Ellipse_Dataset(ds, start_idx=8000)

if args.batch_size is None:
    batch_size = cfg.training_hyperparameters["batch_size"]
else:
    batch_size = args.batch_size
dataloader = DataLoader(pt_ds, num_workers=8, batch_size=batch_size)

model = CNN(cfg)
summary(model, input_size=(batch_size, 1, 128, 128))

model.load_state_dict(torch.load(cfg.model_save_file, weights_only=True))
model.to(device)


print(f"Applying model to {Path(args.data_dir, args.data_file)} \n")
t_time_start = perf_counter()
model.eval()
with torch.no_grad():
    for X, y in dataloader:
        X = X.to(device)
        X = X.view(-1, 1, 128, 128)
        y = y.view(-1)
        pred = model(X).to("cpu")
        try:
            preds = torch.cat((preds, pred), dim=0)
            truth = torch.cat((truth, y), dim=0)
        except NameError:
            preds = pred
            truth = y
t_time = perf_counter() - t_time_start
print(
    f"Done! Total evaluation time: {t_time // 60:.0f} minutes, {t_time % 60:.2f} seconds"
)

print("Saving...")
preds_ds = xr.Dataset(
    data_vars={
        "predictions": (("example", "class"), preds),
        "labels": ("example", truth),
    }
)
preds_ds.to_netcdf(args.save_file)
