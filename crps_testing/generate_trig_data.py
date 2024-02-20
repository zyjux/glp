import numpy as np
import xarray as xr
from pathlib import Path

rng = np.random.default_rng()

samples = []
print('Starting data generation')
for i in range(int(1.5e6)):
    x = rng.uniform(0, 2*np.pi)
    y = rng.uniform(0, 2*np.pi)

    output = np.sin(x) + np.sin(y)

    sample = xr.Dataset(
        {
            "x": x,
            "y": y,
            "target": output
        }
    )
    samples.append(sample)

    if i % 1000 == 0:
        print(f'Processed sample {i}')

training_samples = xr.concat(samples, dim='idx')
print('Saving data to file...')
training_samples.to_netcdf(Path.home() / "research_data/GLP/synthetic_data/trig_test.nc")
print('Done!')