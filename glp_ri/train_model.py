from time import perf_counter

import torch
import torchvision.transforms.v2 as tvtf
from data_utils import (
    DATA_DIR,
    AddGaussianNoise,
    aug_crossentropy_RI_Dataset,
    load_labels,
)
from network_def import CNN, EarlyStopper, crps_loss, train, validate
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchinfo import summary

# PyTorch dropout rate is probability of dropping; TF is probability of retaining
dropout_rate = 0.2

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
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

cnn_train_ds = aug_crossentropy_RI_Dataset(train_labels, transforms=transform_list)
cnn_valid_ds = aug_crossentropy_RI_Dataset(valid_labels)

batches_per_epoch = 32
batch_size = 64
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

cnn_model = CNN(dropout_rate).to(device)
cnn_loss_fn = crps_loss()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, "min")

early_stopper = EarlyStopper(patience=20, min_delta=1e-4)

summary(cnn_model, input_size=(batch_size, 1, 380, 540))

print(
    f"Validation true percentage: {valid_labels[:, -1].sum()/valid_labels.shape[0] * 100}%"
)

epochs = 1000
print("Training CNN \n")
t_time_start = perf_counter()
for t in range(epochs):
    print(f"Epoch {t + 1}\n----------------------------")
    start_time = perf_counter()
    train(cnn_train_dataloader, cnn_model, cnn_loss_fn, cnn_optimizer, device=device)
    val_loss = validate(cnn_valid_dataloader, cnn_model, cnn_loss_fn, device=device)
    scheduler.step(val_loss)
    if early_stopper.early_stop(val_loss):
        print(
            f"Early stopping with minimum validation loss of {early_stopper.min_validation_loss}"
        )
        break
    if early_stopper.counter == 0:
        print(f"Validation loss improved, saving model")
        torch.save(cnn_model.state_dict(), "./saved_models/crps_cnn.pt")

    print(f"Epoch time: {perf_counter() - start_time:.2f} seconds \n")
t_time = perf_counter() - t_time_start
print(
    f"Done! Total training time: {t_time // 60:.0f} minutes, {t_time % 60:.2f} seconds, average epoch time: {t_time/epochs:.2f} seconds"
)
