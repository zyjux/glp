import xarray as xr
import numpy as np
import json
from pathlib import Path

x = np.arange(0, 540)
y = np.arange(0, 380)[:, None]

rng = np.random.default_rng()

prob_square = 0.1
labels = dict()

for storm_id in range(100):
    print(f'Generating storm {storm_id}')
    storm_samples = []
    storm_labels = dict()
    for satellite_valid_time_unix_sec in range(100):
        r = rng.uniform(40, 100)
        x0 = rng.uniform(20, 520)
        y0 = rng.uniform(20, 360)
        square = (rng.uniform(0, 1) <= prob_square)
        if square:
            img = np.logical_and(np.abs(x - x0) <= r, np.abs(y - y0) <= r).astype('float')
        else:
            img = ((x - x0)**2 + (y - y0)**2 <= r**2).astype('float')
        img = xr.DataArray(
            img,
            dims=("y", "x"),
            coords={"y": y[:, 0], "x": x}
        )
        sample = xr.Dataset({"satellite_predictors_gridded": img})
        storm_samples.append(sample)
        storm_labels[str(satellite_valid_time_unix_sec)] = int(square)

    print('Saving...')
    training_samples = xr.concat(storm_samples, dim='satellite_valid_time_unix_sec')
    training_samples.to_netcdf(Path.home() / f"research_data/GLP/synthetic_data/synth_storms/learning_examples_storm{storm_id}.nc")
    labels[f'storm{storm_id}'] = storm_labels

print('Saving labels...')
with open(Path.home() / "research_data/GLP/synthetic_data/synth_storms/train_labels.json", "w") as file:
    json.dump(labels, file, indent=4)