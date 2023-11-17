from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_encoding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.dense_stack = nn.Sequential(
            nn.Linear(8*8*128, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        latent = self.conv_encoding(x)
        latent = self.flatten(latent)
        preds = self.dense_stack(latent)
        return preds, latent


# Define model
class glp_CNN(nn.Module):
    def __init__(self, num_classes, num_angles):
        super().__init__()
        self.full_stack = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 1), padding=(1, 1, 0), padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 1), padding=(1, 1, 0), padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 1), padding=(1, 1, 0), padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 1), padding=(1, 1, 0), padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 1), padding=(1, 1, 0), padding_mode='reflect'),
            nn.ReLU()
        )
        self.glp_step = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 1, 2)),
        )
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Sequential(
            nn.Linear(8*8*128*int(num_angles/2), 100),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x):
        full_latent = self.full_stack(x)
        glp_latent = self.glp_step(full_latent)
        latent = self.flatten(glp_latent)
        latent = self.dense_1(latent)
        preds = self.classifier(latent)
        return preds, full_latent, glp_latent, latent