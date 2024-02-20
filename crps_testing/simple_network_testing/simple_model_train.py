from simple_model import NN, train, validate, mae_loss
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from time import perf_counter
import xarray as xr

DATA_DIR = "/nfs/home/lverho/research_data/GLP/synthetic_data/"

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Create custom dataset method
class SynthDataset(Dataset):
    def __init__(self, full_ds, start_idx=None, end_idx=None, transform=None, target_transform=None):
        self.full_ds = full_ds.isel(idx=slice(start_idx, end_idx))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.full_ds.idx.shape[0]

    def __getitem__(self, idx):
        sample_ds = self.full_ds.isel(idx=idx)
        input = torch.unsqueeze(torch.tensor([sample_ds['x'].item(), sample_ds['y'].item()]), 0)
        label = torch.reshape(torch.tensor(sample_ds['target'].item()), (1, 1))
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            label = self.target_transform(label)
        return input, label


ds = xr.load_dataset(DATA_DIR + 'trig_test.nc')

train_ds = SynthDataset(ds, end_idx=int(1e6))
valid_ds = SynthDataset(ds, start_idx=int(1e6), end_idx=int(1.5e6))

batch_size = 32
train_dataloader = DataLoader(train_ds, num_workers=4, batch_size=batch_size)
valid_dataloader = DataLoader(valid_ds, num_workers=4, batch_size=batch_size)

model = NN().to(device)
loss_fn = mae_loss()
# loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
print('Training NN \n')
t_time_start = perf_counter()
for t in range(epochs):
    print(f"Epoch {t + 1}\n----------------------------")
    start_time = perf_counter()
    train(train_dataloader, model, loss_fn, optimizer, device=device)
    validate(valid_dataloader, model, loss_fn, device=device)
    print(f"Epoch time: {perf_counter() - start_time:.2f} seconds \n")
    torch.save(model.state_dict(), './saved_models/simple_test.pt')
t_time = perf_counter() - t_time_start
print(f"Done! Total training time: {t_time // 60:.0f} minutes, {t_time % 60:.2f} seconds, average epoch time: {t_time/epochs:.2f} seconds")