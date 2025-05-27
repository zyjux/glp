from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import xarray as xr
from synth_network_models import CNN, glp_CNN
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision import transforms


def angle_categorizer(angle, num_categories, diagonal=False):
    region_angle = np.pi / num_categories
    comp_angles = angle / region_angle
    if diagonal:
        comp_angles += 1 / 2
        comp_angles = comp_angles.astype(int) % num_categories
    return comp_angles.astype(int)


# Create custom dataset method
class SynthDataset(Dataset):
    def __init__(
        self,
        full_ds,
        start_idx=None,
        end_idx=None,
        transform=None,
        target_transform=None,
    ):
        self.full_ds = full_ds.isel(p=slice(start_idx, end_idx))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.full_ds.p.shape[0]

    def __getitem__(self, idx):
        image = torch.unsqueeze(
            torch.tensor(self.full_ds.ellipse[idx, :, :].values.astype(np.float32)), 0
        )
        label = torch.tensor(self.full_ds.target.isel(p=idx).values)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


ds = xr.load_dataset(Path.home() / "research_data/GLP/synthetic_data/init_ds.nc")

num_classes = 2
ds["target"] = (("p"), angle_categorizer(ds.angle.values, num_classes, diagonal=True))

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

##################################
###            CNN             ###
##################################

cnn_train_ds = SynthDataset(ds, end_idx=8000)
cnn_val_ds = SynthDataset(ds, start_idx=8000)

batch_size = 32
cnn_train_dataloader = DataLoader(
    cnn_train_ds, num_workers=2, batch_size=batch_size, shuffle=True
)
cnn_val_dataloader = DataLoader(
    cnn_val_ds, num_workers=2, batch_size=batch_size, shuffle=True
)

cnn_model = CNN().to(device)
print(cnn_model)

cnn_loss_fn = nn.CrossEntropyLoss()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-4)
summary(cnn_model, input_size=(batch_size, 1, 128, 128))


def cnn_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred, _ = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 32 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def cnn_validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, _ = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}"
    )


epochs = 10
print("Training CNN \n")
t_time_start = perf_counter()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    start_time = perf_counter()
    cnn_train(cnn_train_dataloader, cnn_model, cnn_loss_fn, cnn_optimizer)
    cnn_validate(cnn_val_dataloader, cnn_model, cnn_loss_fn)
    print(f"Epoch time: {perf_counter() - start_time:.2f} seconds \n")
t_time = perf_counter() - t_time_start
print(
    f"Done! Total training time: {t_time // 60:.0f} minutes, {t_time % 60:.2f} seconds, average epoch time: {t_time/epochs:.2f} seconds"
)

torch.save(cnn_model.state_dict(), "./saved_models/synth_data_network/cnn.pt")


##################################
###          Aug CNN           ###
##################################

aug_cnn_train_ds = SynthDataset(
    ds, end_idx=8000, transform=transforms.RandomRotation(15)
)
aug_cnn_val_ds = SynthDataset(ds, start_idx=8000)

batch_size = 32
aug_cnn_train_dataloader = DataLoader(
    aug_cnn_train_ds, num_workers=2, batch_size=batch_size, shuffle=True
)
aug_cnn_val_dataloader = DataLoader(
    aug_cnn_val_ds, num_workers=2, batch_size=batch_size, shuffle=True
)

aug_cnn_model = CNN().to(device)
print(aug_cnn_model)

aug_cnn_loss_fn = nn.CrossEntropyLoss()
aug_cnn_optimizer = torch.optim.Adam(aug_cnn_model.parameters(), lr=1e-4)

summary(aug_cnn_model, input_size=(batch_size, 1, 128, 128))


def aug_cnn_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred, _ = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 32 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def aug_cnn_validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, _ = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}"
    )


epochs = 10
print("Training Augmented CNN \n")
t_time_start = perf_counter()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    start_time = perf_counter()
    aug_cnn_train(
        aug_cnn_train_dataloader, aug_cnn_model, aug_cnn_loss_fn, aug_cnn_optimizer
    )
    aug_cnn_validate(aug_cnn_val_dataloader, aug_cnn_model, aug_cnn_loss_fn)
    print(f"Epoch time: {perf_counter() - start_time:.2f} seconds \n")
t_time = perf_counter() - t_time_start
print(
    f"Done! Total training time: {t_time // 60:.0f} minutes, {t_time % 60:.2f} seconds, average epoch time: {t_time/epochs:.2f} seconds"
)

torch.save(aug_cnn_model.state_dict(), "./saved_models/synth_data_network/aug_cnn.pt")

##################################
###          GLP CNN           ###
##################################


class glp_transform(nn.Module):
    def __init__(self, num_angles=None, angle_inc=None):
        super().__init__()
        if angle_inc is not None:
            self.num_angles = 360 / angle_inc
            self.angle_inc = angle_inc
        elif num_angles is not None:
            self.angle_inc = 360 / num_angles
            self.num_angles = num_angles
        else:
            raise ValueError("Need either num_angles or angle_inc to be defined")

    def __call__(self, image):
        temp_list = [
            transforms.functional.rotate(image, self.angle_inc * i)
            for i in range(int(self.num_angles))
        ]
        return torch.stack(temp_list, dim=-1)


angle_inc = 30
num_angles = 360 / angle_inc

glp_train_ds = SynthDataset(
    ds, end_idx=8000, transform=glp_transform(angle_inc=angle_inc)
)
glp_val_ds = SynthDataset(
    ds, start_idx=8000, transform=glp_transform(angle_inc=angle_inc)
)

batch_size = 32
glp_train_dataloader = DataLoader(
    glp_train_ds, batch_size=batch_size, num_workers=16, shuffle=True
)
glp_val_dataloader = DataLoader(
    glp_val_ds, batch_size=batch_size, num_workers=16, shuffle=True
)

glp_model = glp_CNN().to(device)
print(glp_model)

glp_loss_fn = nn.CrossEntropyLoss()
glp_optimizer = torch.optim.Adam(glp_model.parameters(), lr=1e-4)

summary(glp_model, input_size=(batch_size, 1, 128, 128, int(num_angles)))


def glp_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred, _, _, _ = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 32 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def glp_validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            (
                pred,
                _,
                _,
                _,
            ) = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}"
    )


epochs = 10
print("Training GLP CNN \n")
t_time_start = perf_counter()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    start_time = perf_counter()
    glp_train(glp_train_dataloader, glp_model, glp_loss_fn, glp_optimizer)
    glp_validate(glp_val_dataloader, glp_model, glp_loss_fn)
    print(f"Epoch time: {perf_counter() - start_time:.2f} seconds \n")
t_time = perf_counter() - t_time_start
print(
    f"Done! Total training time: {t_time // 60:.0f} minutes, {t_time % 60:.2f} seconds, average epoch time: {t_time/epochs:.2f} seconds"
)

torch.save(glp_model.state_dict(), "./saved_models/synth_data_network/glp.pt")
