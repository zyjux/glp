from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from torch import nn
from torchvision.transforms.functional import rotate


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


class glp_rotation_stack(nn.Module):
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
            rotate(image, self.angle_inc * i) for i in range(int(self.num_angles))
        ]
        return torch.stack(temp_list, dim=-1)


class glp_rotation_pool(nn.Module):
    def __init__(self, glp_pooling_size: int, **kwargs):
        super().__init__()
        self.kernel_size = glp_pooling_size

    def __call__(self, image):
        num_angles = image.shape[-1]
        angle_increment = 360 / num_angles
        output_image = torch.empty(image.shape).to(image.device)
        for i in range(num_angles):
            relative_offset = i % self.kernel_size
            output_image[..., i] = rotate(
                image[..., i], -angle_increment * relative_offset
            )
        return nn.MaxPool3d(kernel_size=(1, 1, self.kernel_size))(output_image)


def conv_block(in_channels, out_channels, leaky_relu_slope, kernel_size, pooling_size):
    padding = int(kernel_size / 2)
    layers = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="reflect",
        ),
        nn.LeakyReLU(leaky_relu_slope),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="reflect",
        ),
        nn.LeakyReLU(leaky_relu_slope),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=pooling_size),
    )
    return layers


def glp_conv_block(
    in_channels, out_channels, leaky_relu_slope, kernel_size, pooling_size
):
    padding = int(kernel_size / 2)
    layers = nn.Sequential(
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size, 1),
            padding=(padding, padding, 0),
            padding_mode="reflect",
        ),
        nn.LeakyReLU(leaky_relu_slope),
        nn.BatchNorm3d(out_channels),
        nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size, 1),
            padding=(padding, padding, 0),
            padding_mode="reflect",
        ),
        nn.LeakyReLU(leaky_relu_slope),
        nn.BatchNorm3d(out_channels),
        nn.MaxPool3d(kernel_size=(pooling_size, pooling_size, 1)),
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


@dataclass
class model_config:
    model_save_file: str
    training_hyperparameters: dict
    data_augmentation: dict = field(default_factory=dict)
    setup_layers: dict = field(default_factory=dict)
    encoding_layers: dict = field(default_factory=dict)
    encoding_layer_defaults: dict = field(default_factory=dict)
    dense_layers: dict = field(default_factory=dict)
    dense_layer_defaults: dict = field(default_factory=dict)
    output_layers: dict = field(default_factory=dict)

    def apply_defaults(self, layers: dict, defaults: dict) -> dict:
        new_layers = layers.copy()
        for layer_name in new_layers:
            for default_name in defaults:
                if default_name not in new_layers[layer_name]:
                    new_layers[layer_name][default_name] = defaults[default_name]
        return new_layers

    def translate_layers(self, layers: dict) -> OrderedDict:
        translated_layers = OrderedDict(
            [
                (
                    layer_name,
                    self.layer_name_translation[layers[layer_name].pop("type")](
                        **layers[layer_name]
                    ),
                )
                for layer_name in layers
            ]
        )
        return translated_layers

    def __post_init__(self):
        self.layer_name_translation = {
            "conv_block": conv_block,
            "dense_block": dense_block,
            "linear": nn.Linear,
            "glp_rotation_stack": glp_rotation_stack,
            "glp_rotation_pool": glp_rotation_pool,
            "glp_conv_block": glp_conv_block,
        }

        self.encoding_layers = self.apply_defaults(
            self.encoding_layers, self.encoding_layer_defaults
        )
        self.dense_layers = self.apply_defaults(
            self.dense_layers, self.dense_layer_defaults
        )
        self.setup_layers_trans = self.translate_layers(self.setup_layers)
        self.encoding_layers_trans = self.translate_layers(self.encoding_layers)
        self.dense_layers_trans = self.translate_layers(self.dense_layers)
        self.output_layers_trans = self.translate_layers(self.output_layers)


class CNN(nn.Module):
    def __init__(self, cfg: model_config):
        super().__init__()

        self.setup_layers = nn.Sequential(cfg.setup_layers_trans)
        self.encoding_layers = nn.Sequential(cfg.encoding_layers_trans)
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(cfg.dense_layers_trans)
        self.output_layers = nn.Sequential(cfg.output_layers_trans)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        setup_out = self.setup_layers(x)
        encoding_out = self.encoding_layers(setup_out)
        flattened_out = self.flatten(encoding_out)
        dense_out = self.dense_layers(flattened_out)
        logit_out = self.output_layers(dense_out)
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
