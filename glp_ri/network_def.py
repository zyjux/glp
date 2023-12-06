import torch
from torch import nn


class crps_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction_f = {
            'none': nn.Identity(),
            'mean': torch.mean,
            'sum': torch.sum,
        }.get(reduction)

    def forward(self, predicted, target):
        # Unsqueeze the target vector to allow each value to be broadcast against the N predictors for each case
        mean_prediction_error_tensor = torch.mean(torch.abs(predicted - torch.unsqueeze(target, -1)), dim=-1)

        # Unsqueeze the predictors in different spots so that they broadcast and cross-compare
        mean_prediction_var_tensor = torch.mean(torch.abs(
            torch.unsqueeze(predicted, -1) - torch.unsqueeze(predicted, -2)
        ), dim=(-1, -2))

        # Use the reduction function we got from __init__ to reduce to a single number (or not if reduction == 'none')
        return self.reduction_f(mean_prediction_error_tensor - 0.5 * mean_prediction_var_tensor)


def conv_block(in_channels, out_channels):
    layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    return layers


class CNN(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()

        self.conv_layers = nn.Sequential(
            conv_block(1, 8),
            conv_block(8, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256)
        )
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(10240, 1740),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1740, 1305),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1305, 870),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(870, 435),
            nn.ReLU(),
            nn.Linear(435, 200)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = self.flatten(conv_out)
        dense_out = self.dense_layers(conv_out)
        dense_out = torch.reshape(dense_out, (-1, 100, 2))
        soft_out = self.softmax(dense_out)[:, :, 0]
        return soft_out


def train(dataloader, model, loss_fn, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 24 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss: >7f} [{current: >5d}/{size:>5d}]")


def validate(dataloader, model, loss_fn, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += (pred.mean(axis=-1) >= 0.5).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")