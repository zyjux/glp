from time import perf_counter

import torch
import xarray as xr
from data_utils import DATA_DIR, aug_crossentropy_RI_Dataset, load_labels
from network_def import CNN
from torch.utils.data import DataLoader
from torchinfo import summary

# PyTorch dropout rate is probability of dropping; TF is probability of retaining
dropout_rate = 0.2

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

valid_labels, valid_weights = load_labels(
    DATA_DIR + "/valid_labels.json", desired_ratio=None
)

cnn_valid_ds = aug_crossentropy_RI_Dataset(valid_labels)

batch_size = 64
cnn_valid_dataloader = DataLoader(cnn_valid_ds, num_workers=8, batch_size=batch_size)

cnn_model = CNN(dropout_rate).to(device)

summary(cnn_model, input_size=(batch_size, 1, 380, 540))

print(
    f"Validation true percentage: {valid_labels[:, -1].sum()/valid_labels.shape[0] * 100}%"
)

print("Applying CNN \n")
t_time_start = perf_counter()
cnn_model.eval()
with torch.no_grad():
    for X, y in cnn_valid_dataloader:
        X = X.to(device)
        X = X.view(-1, 1, 380, 540)
        y = y.view(-1)
        pred = cnn_model(X).to("cpu")
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
        "predictions": (("example", "ens_member"), preds),
        "labels": ("example", truth),
    }
)
preds_ds.to_netcdf("~/glp/glp_ri/data/crps_cnn_validation_preds.nc")
