from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

cnn_files = [
    Path("outputs", "cnn_validation_" + ratio + ".nc")
    for ratio in ["1_0", "0_5", "0_25", "0_1", "0_05", "0_01", "0_005"]
]
glp_files = [
    Path("outputs", "glp_validation_" + ratio + ".nc")
    for ratio in ["1_0", "0_5", "0_25", "0_1", "0_05", "0_01", "0_005"]
]

cnn_accuracies = []
for fn in cnn_files:
    ds = xr.open_dataset(fn)
    mean_preds = ds["predictions"].mean(dim="class")
    true_positive_count = (
        np.logical_and(ds["labels"] == 1, mean_preds.round() == 1).sum().item()
    )
    true_negative_count = (
        np.logical_and(ds["labels"] == 0, mean_preds.round() == 0).sum().item()
    )
    accuracy = (true_positive_count + true_negative_count) * 100 / ds.sizes["example"]
    cnn_accuracies.append(accuracy)

glp_accuracies = []
for fn in glp_files:
    ds = xr.open_dataset(fn)
    mean_preds = ds["predictions"].mean(dim="class")
    true_positive_count = (
        np.logical_and(ds["labels"] == 1, mean_preds.round() == 1).sum().item()
    )
    true_negative_count = (
        np.logical_and(ds["labels"] == 0, mean_preds.round() == 0).sum().item()
    )
    accuracy = (true_positive_count + true_negative_count) * 100 / ds.sizes["example"]
    glp_accuracies.append(accuracy)

ratios = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]

F, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_xscale("log")
ax.plot(ratios, cnn_accuracies, label="CNN")
ax.plot(ratios, glp_accuracies, label="GLP")
ax.xaxis.set_inverted(True)
ax.legend()

F.savefig("accuracy_curve.png", bbox_inches="tight", dpi=300)
