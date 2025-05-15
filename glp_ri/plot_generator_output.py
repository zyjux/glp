from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2 as tvtf
from data_utils import (DATA_DIR, AddGaussianNoise, load_labels,
                        verbose_aug_crossentropy_RI_Dataset)
from torch.utils.data import DataLoader, WeightedRandomSampler

# PyTorch dropout rate is probability of dropping; TF is probability of retaining
dropout_rate = 0.2

# Get cpu, gpu or mps device for training.
device = "cpu"
print(f"Using {device} device")

train_labels, train_weights = load_labels(
    DATA_DIR + "/train_labels.json", desired_ratio=(3, 1)
)
valid_labels, valid_weights = load_labels(
    DATA_DIR + "/valid_labels.json", desired_ratio=(1, 1)
)

rotate_transform = tvtf.RandomRotation(50)  # type: ignore
noise_transform = AddGaussianNoise(std=0.5)
translate_transform = tvtf.RandomAffine(0, translate=(0.05, 0.05))  # type: ignore

transform_list = (
    [rotate_transform] * 2 + [translate_transform] * 2 + [noise_transform] * 3
)

cnn_train_ds = verbose_aug_crossentropy_RI_Dataset(
    train_labels, transforms=transform_list
)
cnn_valid_ds = verbose_aug_crossentropy_RI_Dataset(valid_labels)

batches_per_epoch = 3
batch_size = 16
wtd_sampler = WeightedRandomSampler(
    train_weights, batches_per_epoch * batch_size, replacement=False  # type: ignore
)
valid_sampler = WeightedRandomSampler(
    valid_weights, batches_per_epoch * batch_size, replacement=False  # type: ignore
)
cnn_train_dataloader = DataLoader(
    cnn_train_ds, num_workers=8, batch_size=batch_size, sampler=wtd_sampler
)
cnn_valid_dataloader = DataLoader(
    cnn_valid_ds, num_workers=8, batch_size=batch_size, sampler=valid_sampler
)

print(
    f"Validation true percentage: {valid_labels[:, -1].sum()/valid_labels.shape[0] * 100}%"
)

print("Drawing training samples")
t_time_start = perf_counter()
for batch_id, (X, y, storm_id, timestamp) in enumerate(cnn_train_dataloader):
    true_x_len = X.shape[0]

    F = plt.figure(constrained_layout=True, figsize=(12, 12))
    axes = F.subplots(
        int(np.ceil(np.sqrt(true_x_len))), int(np.ceil(np.sqrt(true_x_len)))
    )
    for i, ax in enumerate(np.ravel(axes)):  # type: ignore
        ax.imshow(X[i, 0, 0, :, :])
        if y[i, 0] == 1:
            ax.patch.set_linewidth(5)
            ax.patch.set_edgecolor("red")
        this_storm_id = storm_id[0][i]
        this_timestamp = timestamp[0][i]
        this_timestamp = datetime.fromtimestamp(float(this_timestamp), tz=None)
        ax.set_title(f"{this_storm_id}, {this_timestamp}")

    F.savefig(f"figures/generator_train_batch_{batch_id}.png")

print("Drawing validation samples")
t_time_start = perf_counter()
for batch_id, (X, y, storm_id, timestamp) in enumerate(cnn_valid_dataloader):
    true_x_len = X.shape[0]

    F = plt.figure(constrained_layout=True, figsize=(12, 12))
    axes = F.subplots(
        int(np.ceil(np.sqrt(true_x_len))), int(np.ceil(np.sqrt(true_x_len)))
    )
    for i, ax in enumerate(np.ravel(axes)):  # type: ignore
        ax.imshow(X[i, 0, 0, :, :])
        if y[i, 0] == 1:
            ax.patch.set_linewidth(5)
            ax.patch.set_edgecolor("red")
        this_storm_id = storm_id[0][i]
        this_timestamp = timestamp[0][i]
        this_timestamp = datetime.fromtimestamp(float(this_timestamp), tz=None)
        ax.set_title(f"{this_storm_id}, {this_timestamp}")

    F.savefig(f"figures/generator_validation_batch_{batch_id}.png")
