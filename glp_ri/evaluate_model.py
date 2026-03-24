import argparse
from pathlib import Path

import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(prog="EvalModel", description="Evaluates models")
parser.add_argument("experiment_dir", type=Path)
parser.add_argument("-f", "--model_files", nargs="+", type=Path)
args = parser.parse_args()

# Open predictions dataset
model_files = [Path(args.experiment_dir, file) for file in args.model_files]

for model_file in model_files:
    ds = xr.open_dataset(model_file)
    print(ds["predictions"].shape)
    #### NEED TO UPDATE THIS TO WORK FOR CRPS OUTPUTS

    # Get deterministic mean predictions
    mean_preds = ds["predictions"][:, 1]

    print(f"\nEvaluating {model_file}")

    # Dataset info
    print(f"\nDataset information:\n----------")
    print(f"Total examples: {ds.sizes['example']}")
    print(f"True positive cases: {ds['labels'].sum().item()}")
    print(f"True positive ratio: {ds['labels'].sum().item() / ds.sizes['example']:.3f}")

    # Compute positive prediction ratio
    positive_ratio = mean_preds.round().sum() / ds.sizes["example"]
    print("\nRaw prediction info:\n----------")
    print(f"Positive prediction ratio: {positive_ratio:.3f}")

    # Confusion matrix
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

    # Accuracy
    print("\nAccuracy\n----------")
    accuracy = (true_positive_count + true_negative_count) * 100 / ds.sizes["example"]
    print(f"Accuracy: {accuracy:.2f}%")

    # Crossentropy
    print("\nCrossentropy\n----------")
    crossentropy = (
        -1
        * (
            ds["labels"] * np.log(mean_preds + 1e-7)
            + (1 - ds["labels"]) * np.log(1 - mean_preds + 1e-7)
        ).mean()
    )
    print(f"Binary crossentropy: {crossentropy:.4f}")

    print("\nEvaluations complete.\n \n")
