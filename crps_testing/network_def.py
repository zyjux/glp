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
        mean_prediction_error_tensor = torch.mean(torch.abs(predicted - target), dim=-1)

        # Unsqueeze the predictors in different spots so that they broadcast and cross-compare
        mean_prediction_var_tensor = torch.mean(torch.abs(
            torch.unsqueeze(predicted, -1) - torch.unsqueeze(predicted, -2)
        ), dim=(-1, -2))

        # Use the reduction function we got from __init__ to reduce to a single number (or not if reduction == 'none')
        return self.reduction_f(mean_prediction_error_tensor - 0.5 * mean_prediction_var_tensor)


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_layers = nn.Sequential(
            nn.Linear(2, 15),
            nn.ReLU(),
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, 60),
            nn.ReLU(),
            nn.Linear(60, 100),
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