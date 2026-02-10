from collections import OrderedDict

import torch
from torch import nn


class crps_loss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction_f = {
            "none": nn.Identity(),
            "mean": torch.mean,
            "sum": torch.sum,
        }.get(reduction)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        return self.reduction_f(
            torch.mean(torch.abs(predicted - torch.unsqueeze(target, dim=-1)), dim=-1)
            - 0.5
            * torch.mean(
                torch.abs(
                    torch.unsqueeze(predicted, -1) - torch.unsqueeze(predicted, -2)
                ),
                dim=(-1, -2),
            )
        )  # type: ignore


def conv_block(in_channels, out_channels, leaky_relu_slope):
    layers = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        ),
        nn.LeakyReLU(leaky_relu_slope),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        ),
        nn.LeakyReLU(leaky_relu_slope),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2),
    )
    return layers


def dense_block(in_neurons, out_neurons, dropout_rate, leaky_relu_slope):
    layers = nn.Sequential(
        nn.Linear(in_neurons, out_neurons),
        nn.LeakyReLU(leaky_relu_slope),
        nn.Dropout(dropout_rate),
        nn.BatchNorm1d(out_neurons),
    )
    return layers


"""
conv_block(1, 8),
conv_block(8, 16),
conv_block(16, 32),
conv_block(32, 64),
conv_block(64, 128),
conv_block(128, 256),

            OrderedDict(
                [
                    (
                        f"dense_block_{i}",
                        dense_block(
                            dense_neurons[i],
                            dense_neurons[i + 1],
                            dropout_rate,
                            LEAKY_RELU_SLOPE,
                        ),
                    )
                    for i in range(len(conv_channels) - 2)
                ]
                + [("output_layer", nn.Linear(dense_neurons[-2], dense_neurons[-1]))]
            )
            dense_block(10240, 1740, dropout_rate, LEAKY_RELU_SLOPE),
            dense_block(1740, 1305, dropout_rate, LEAKY_RELU_SLOPE),
            dense_block(1305, 870, dropout_rate, LEAKY_RELU_SLOPE),
            dense_block(870, 435, dropout_rate, LEAKY_RELU_SLOPE),
            nn.Linear(435, 100),
"""

LAYER_NAME_TRANSLATION = {
    "conv_block": conv_block,
    "dense_block": dense_block,
    "linear": nn.Linear,
}


class CNN(nn.Module):
    def __init__(
        self,
        encoding_layers: dict,
        dense_layers: dict,
        output_layer: dict,
        encoding_layer_defaults: dict,
        dense_layer_defaults: dict,
    ):
        super().__init__()

        for layer_name in encoding_layers:
            for default_name in encoding_layer_defaults:
                if default_name not in encoding_layers[layer_name]:
                    encoding_layers[layer_name][default_name] = encoding_layer_defaults[
                        default_name
                    ]
        for layer_name in dense_layers:
            for default_name in dense_layer_defaults:
                if default_name not in dense_layers[layer_name]:
                    dense_layers[layer_name][default_name] = dense_layer_defaults[
                        default_name
                    ]

        translated_encoding_layers = OrderedDict(
            [
                (
                    layer_name,
                    LAYER_NAME_TRANSLATION[encoding_layers[layer_name].pop("type")](
                        **encoding_layers[layer_name]
                    ),
                )
                for layer_name in encoding_layers
            ]
        )

        translated_dense_layers = OrderedDict(
            [
                (
                    layer_name,
                    LAYER_NAME_TRANSLATION[dense_layers[layer_name].pop("type")](
                        **dense_layers[layer_name]
                    ),
                )
                for layer_name in dense_layers
            ]
        )

        translated_output_layer = OrderedDict(
            [
                (
                    layer_name,
                    LAYER_NAME_TRANSLATION[output_layer[layer_name].pop("type")](
                        **output_layer[layer_name]
                    ),
                )
                for layer_name in output_layer
            ]
        )

        self.conv_layers = nn.Sequential(translated_encoding_layers)
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(translated_dense_layers)
        self.output_layer = nn.Sequential(translated_output_layer)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = self.flatten(conv_out)
        dense_out = self.dense_layers(conv_out)
        logit_out = self.output_layer(dense_out)
        sig_out = self.sigmoid(logit_out)
        return sig_out


def train(dataloader, model, loss_fn, optimizer, accumulation_batches=1, device="cpu"):
    size = len(dataloader) * dataloader.batch_size
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        true_x_len = X.shape[0]
        X = X.view(-1, 1, 380, 540)
        y = y.view(-1)
        print(f"Batch contains {torch.sum(y)} positive examples out of {X.shape[0]}")
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss = loss / accumulation_batches

        loss.backward()

        if ((batch + 1) % accumulation_batches == 0) or (batch + 1 == len(dataloader)):
            print("Optimization step")
            optimizer.step()
            optimizer.zero_grad()

        if batch % 2 == 0:
            loss, current = loss.item(), (batch + 1) * true_x_len
            print(f"loss: {loss: >7f} [{current: >5d}/{size:>5d}]")


def validate(dataloader, model, loss_fn, device="cpu"):
    size = len(dataloader) * dataloader.batch_size
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, positive_preds = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(-1, 1, 380, 540)
            y = y.view(-1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            positive_preds += (
                ((torch.round(pred.mean(axis=-1))).type(torch.float))
                .sum()
                .item()
                # torch.argmax(pred, dim=-1).type(torch.float).sum().item()
            )
            correct += (
                ((torch.round(pred.mean(axis=-1)) == y).type(torch.float))
                .sum()
                .item()
                # (torch.argmax(pred, dim=-1) == y).type(torch.float).sum().item()
            )
    test_loss /= num_batches
    correct /= size
    positive_preds /= size
    print(
        f"Validation Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, Positive ratio: {positive_preds:>6f}"
    )
    return test_loss, correct


class EarlyStopper:
    """Implements early stopping when validation loss hasn't improved significantly"""

    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.max_accuracy = 0.0

    def early_stop(self, validation_loss, accuracy):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.max_accuracy = accuracy
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
