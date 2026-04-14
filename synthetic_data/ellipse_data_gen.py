from pathlib import Path

import numpy as np
import scipy.ndimage as nd
import xarray as xr

# Set up random number generator and axes
rng = np.random.default_rng()
x = np.arange(0, 128)
y = np.arange(0, 128)[:, None]

# Generate training data
samples = []
for i in range(10000):
    maj_len = rng.uniform(10.0, 40.0)
    min_len = rng.uniform(5.0, maj_len / 1.5)
    ang = rng.uniform(0, np.pi)
    (x0, y0) = rng.uniform(20.0, 108, 2)
    ellipse = (
        ((x - x0) * np.cos(ang) + (y - y0) * np.sin(ang)) / (maj_len / 2)
    ) ** 2 + (
        ((x - x0) * np.sin(ang) - (y - y0) * np.cos(ang)) / (min_len / 2)
    ) ** 2 <= 1
    ellipse = xr.DataArray(
        np.exp(-0.1 * nd.distance_transform_edt(1 - ellipse)),  # type: ignore
        dims=("x", "y"),
        coords={"x": x, "y": x},
    )
    sample = xr.Dataset(
        {
            "ellipse": ellipse,
            "maj_len": maj_len,
            "min_len": min_len,
            "angle": ang,
            "center_x": x0,
            "center_y": y0,
        }
    )
    samples.append(sample)
training_samples = xr.concat(samples, dim="sample")
training_samples.to_netcdf(
    Path("/mnt/data2/lverhoef/synthetic_ellipses/train_valid_ds.nc"), mode="w"
)
