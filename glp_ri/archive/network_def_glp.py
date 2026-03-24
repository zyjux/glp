import torch
from torch import nn
from torchvision.transforms.functional import rotate

LEAKY_RELU_SLOPE = 0.2


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
            nn.Linear(435, 100),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        expanded = self.expand_symmetries(x)
        conv_out = self.conv_layers(expanded)
        conv_out = self.flatten(conv_out)
        dense_out = self.dense_layers(conv_out)
        sig_out = self.sigmoid(dense_out)
        return sig_out
