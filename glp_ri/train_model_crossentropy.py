from data_utils import DATA_DIR, load_labels, crossentropy_RI_Dataset, aug_crossentropy_RI_Dataset, AddGaussianNoise
from network_def_crossentropy import CNN_direct, train, validate
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as tvtf
from torchinfo import summary
from time import perf_counter

# PyTorch dropout rate is probability of dropping; TF is probability of retaining
dropout_rate = 0.0

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

train_labels, train_weights = load_labels(DATA_DIR + '/train_labels.json')
valid_labels, valid_weights = load_labels(DATA_DIR + '/valid_labels.json')

rotate_transform = tvtf.RandomRotation(50)
noise_transform = AddGaussianNoise(std=0.5)
translate_transform = tvtf.RandomAffine(0, translate=(0.05, 0.05))

transform_list = [rotate_transform]*2 + [translate_transform]*2 + [noise_transform]*3

# cnn_train_ds = aug_crossentropy_RI_Dataset(train_labels, transforms=transform_list)
# cnn_valid_ds = aug_crossentropy_RI_Dataset(valid_labels)
cnn_train_ds = crossentropy_RI_Dataset(train_labels)
cnn_valid_ds = crossentropy_RI_Dataset(valid_labels)

batches_per_epoch = 32
batch_size = 16
wtd_sampler = WeightedRandomSampler(train_weights, batches_per_epoch*batch_size, replacement=True)
valid_wtd_sampler = WeightedRandomSampler(valid_weights, batches_per_epoch*batch_size, replacement=True)
cnn_train_dataloader = DataLoader(cnn_train_ds, num_workers=8, batch_size=batch_size, sampler=wtd_sampler)
cnn_valid_dataloader = DataLoader(cnn_valid_ds, num_workers=8, batch_size=batch_size, sampler=valid_wtd_sampler)

cnn_model = CNN_direct().to(device)
cnn_loss_fn = nn.CrossEntropyLoss()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-5)

summary(cnn_model)

epochs = 1000
print('Training CNN \n')
t_time_start = perf_counter()
for t in range(epochs):
    print(f"Epoch {t + 1}\n----------------------------")
    start_time = perf_counter()
    train(cnn_train_dataloader, cnn_model, cnn_loss_fn, cnn_optimizer, device=device)
    validate(cnn_valid_dataloader, cnn_model, cnn_loss_fn, device=device)
    print(f"Epoch time: {perf_counter() - start_time:.2f} seconds \n")
    torch.save(cnn_model.state_dict(), './saved_models/crossentropy_cnn.pt')
t_time = perf_counter() - t_time_start
print(f"Done! Total training time: {t_time // 60:.0f} minutes, {t_time % 60:.2f} seconds, average epoch time: {t_time/epochs:.2f} seconds")