import torch
from torch import nn

LEAKY_RELU_SLOPE = 0.2


class crps_loss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction_f = {
            "none": nn.Identity(),
            "mean": torch.mean,
            "sum": torch.sum,
        }.get(reduction)

    def forward(self, predicted, target):
        return self.reduction_f(
            torch.mean(torch.abs(predicted - torch.unsqueeze(target, dim=-1)), dim=-1)
            - 0.5
            * torch.mean(
                torch.abs(
                    torch.unsqueeze(predicted, -1) - torch.unsqueeze(predicted, -2)
                ),
                dim=(-1, -2),
            )
        )


def conv_block(in_channels, out_channels):
    layers = nn.Sequential(
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 1),
            padding=(1, 1, 0),
            padding_mode="reflect",
        ),
        nn.LeakyReLU(LEAKY_RELU_SLOPE),
        nn.BatchNorm3d(out_channels),
        nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 1),
            padding=(1, 1, 0),
            padding_mode="reflect",
        ),
        nn.LeakyReLU(LEAKY_RELU_SLOPE),
        nn.BatchNorm3d(out_channels),
        nn.MaxPool3d(kernel_size=(2, 2, 1)),
    )
    return layers


class glp_CNN(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()

        self.conv_layers = nn.Sequential(
            conv_block(1, 8),
            conv_block(8, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(10240, 1740),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(1740),
            nn.Linear(1740, 1305),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(1305),
            nn.Linear(1305, 870),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(870),
            nn.Linear(870, 435),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(435),
            nn.Linear(435, 100),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = self.flatten(conv_out)
        dense_out = self.dense_layers(conv_out)
        sig_out = self.sigmoid(dense_out)
        return sig_out


def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader) * dataloader.batch_size
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        true_x_len = X.shape[0]
        X = X.view(-1, 1, 380, 540)
        y = y.view(-1)
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 2 == 0:
            loss, current = loss.item(), (batch + 1) * true_x_len
            print(f"loss: {loss: >7f} [{current: >5d}/{size:>5d}]")


def validate(dataloader, model, loss_fn, device="cpu"):
    size = len(dataloader) * dataloader.batch_size
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(-1, 1, 380, 540)
            y = y.view(-1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (
                ((torch.round(pred.mean(axis=-1)) == y).type(torch.float)).sum().item()
            )
    test_loss /= num_batches
    correct /= size
    print(
        f"Validation Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}"
    )
