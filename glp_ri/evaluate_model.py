import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Open predictions dataset
ds = xr.open_dataset("~/glp/glp_ri/data/crps_cnn_validation_preds.nc")

# Get deterministic mean predictions
mean_preds = ds["predictions"].mean(dim="ens_member")

# Dataset info
print(f"\nDataset information:\n----------")
print(f"Total examples: {ds.sizes['example']}")
print(f"True positive cases: {ds['labels'].sum().item()}")
print(f"True positive ratio: {ds['labels'].sum().item() / ds.sizes['example']:.3f}")

# Compute positive prediction ratio
positive_ratio = mean_preds.round().sum() / ds.sizes["example"]
print("\nRaw prediction info:\n----------")
print(f"Positive prediction ratio: {positive_ratio:.3f}")

# Accuracy
print("\nConfusion Matrix\n----------")
true_positive_count = (
    np.logical_and(ds["labels"] == 1, mean_preds.round() == 1).sum().item()
)
false_positive_count = (
    np.logical_and(ds["labels"] == 0, mean_preds.round() == 1).sum().item()
)
false_negative_count = (
    np.logical_and(ds["labels"] == 1, mean_preds.round() == 0).sum().item()
)
true_negative_count = (
    np.logical_and(ds["labels"] == 0, mean_preds.round() == 0).sum().item()
)
print(f"TP: {true_positive_count} | FP: {false_positive_count}")
print(f"FN: {false_negative_count} | TN: {true_negative_count}")

# Crossentropy
print("\nCrossentropy\n----------")
crossentropy = (
    -1
    * (
        ds["labels"] * np.log(mean_preds) + (1 - ds["labels"]) * np.log(1 - mean_preds)
    ).mean()
)
print(f"Binary crossentropy: {crossentropy:.4f}")

# Create spread-skill plot
individual_crossentropies = -1 * (
    (ds["labels"] * np.log(mean_preds)) + ((1 - ds["labels"]) * np.log(1 - mean_preds))
)
individual_mean_errors = np.abs(mean_preds - ds["labels"])
spreads = np.sqrt(((ds["predictions"] - mean_preds) ** 2).mean(dim="ens_member"))

F = plt.figure(constrained_layout=True, figsize=(5, 5))
ax = plt.gca()
ax.scatter(spreads, individual_crossentropies)
ax.set_xlabel("RMS Spread")
ax.set_ylabel("Crossentropy")
ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
F.savefig("./figures/crps_spread_skill.png")

F = plt.figure(constrained_layout=True, figsize=(5, 5))
ax = plt.gca()
ax.scatter(mean_preds, individual_mean_errors)
ax.set_ylim(0, 1)
ax.set_xlabel("Prediction")
ax.set_ylabel("RMS spread")
F.savefig("./figures/crps_spread_pred.png")

print("\nEvaluations complete.\n \n")
