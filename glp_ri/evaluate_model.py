import numpy as np
import xarray as xr

# Open predictions dataset
model_names = ["crps", "glp"]

for model_name in model_names:
    ds = xr.open_dataset(f"~/glp/glp_ri/data/{model_name}_cnn_validation_preds.nc")

    # Get deterministic mean predictions
    mean_preds = ds["predictions"][:, 1]

    print(f"\nModel type: {model_name}")

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
            ds["labels"] * np.log(mean_preds)
            + (1 - ds["labels"]) * np.log(1 - mean_preds)
        ).mean()
    )
    print(f"Binary crossentropy: {crossentropy:.4f}")

    print("\nEvaluations complete.\n \n")
