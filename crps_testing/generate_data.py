import numpy as np
import xarray as xr
from pathlib import Path

rng = np.random.default_rng()

samples = []
print('Starting data generation')
for i in range(int(1e6)):
    true_mean = rng.uniform(-10, 10)
    true_sigma = rng.uniform(0.5, 1.5)

    for j in range(10):
        output = rng.normal(true_mean, true_sigma)

        sample = xr.Dataset(
            {
                "mean": true_mean,
                "sig": true_sigma,
                "target": output
            }
        )
        samples.append(sample)

    if i % 1000 == 0:
        print(f'Processed sample {i}')

training_samples = xr.concat(samples, dim='idx')
print('Saving data to file...')
training_samples.to_netcdf(Path.home() / "research_data/GLP/synthetic_data/crps_test.nc")
print('Done!')