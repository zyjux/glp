from data_utils import DATA_DIR, load_labels, RI_Dataset
from network_def import CNN, train, validate, crps_loss
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from time import perf_counter

dropout_rate = 0.8

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
valid_labels, _ = load_labels(DATA_DIR + '/valid_labels.json')

cnn_train_ds = RI_Dataset(train_labels)
cnn_valid_ds = RI_Dataset(valid_labels)

batch_size = 16*8
wtd_sampler = WeightedRandomSampler(train_weights, len(train_labels)*8, replacement=True)
cnn_train_dataloader = DataLoader(cnn_train_ds, num_workers=2, batch_size=batch_size, sampler=wtd_sampler)
cnn_valid_dataloader = DataLoader(cnn_valid_ds, num_workers=2, batch_size=batch_size)

cnn_model = CNN().to(device)
cnn_loss_fn = crps_loss()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)

epochs = 10
print('Training CNN \n')
t_time_start = perf_counter()
for t in range(epochs):
    print(f"Epoch {t + 1}\n----------------------------")
    start_time = perf_counter()
    train(cnn_train_dataloader, cnn_model, cnn_loss_fn, cnn_optimizer, device=device)
    validate(cnn_valid_dataloader, cnn_model, cnn_loss_fn, device=device)
    print(f"Epoch time: {perf_counter() - start_time:.2f} seconds \n")
t_time = perf_counter() - t_time_start
print(f"Done! Total training time: {t_time // 60:.0f} minutes, {t_time % 60:.2f} seconds, average epoch time: {t_time/epochs:.2f} seconds")

torch.save(cnn_model.state_dict(), './saved_models/synth_data_network/cnn.pt')