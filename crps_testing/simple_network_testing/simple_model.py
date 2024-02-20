import torch
from torch import nn


class mae_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        return torch.mean(torch.abs(predicted - target))


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_layers = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.hidden_layers(x)


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

        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss: >7f} [{current: >5d}/{size:>5d}]")


def validate(dataloader, model, loss_fn, device='cpu'):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Validation avg loss: {test_loss:>8f}")