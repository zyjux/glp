import torch
from torch import nn
from torchvision.transforms.functional import rotate

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
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

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


def spatial_conv_block(in_channels, out_channels):
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


class glp_rotation_CNN(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()

        self.expand_symmetries = glp_rotation_stack(angle_inc=30)
        self.conv_layers = nn.Sequential(
            spatial_conv_block(1, 8),
            spatial_conv_block(8, 16),
            spatial_conv_block(16, 32),
            glp_rotation_pool(2),
            spatial_conv_block(32, 64),
            spatial_conv_block(64, 128),
            spatial_conv_block(128, 256),
        )
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(61440, 1740),
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
            nn.Linear(435, 2),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        expanded = self.expand_symmetries(x)
        conv_out = self.conv_layers(expanded)
        conv_out = self.flatten(conv_out)
        dense_out = self.dense_layers(conv_out)
        # sig_out = self.sigmoid(dense_out)
        return dense_out


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
            correct += (torch.round(pred) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Validation Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}"
    )


class EarlyStopper:
    """Implements early stopping when validation loss hasn't improved significantly"""

    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
