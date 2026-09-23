import torch
import torch.nn as nn


class BaselineCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(

            # 3 -> 32
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # 32 -> 64
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # 64 -> 128
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # 128 -> 128
            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # Density regression head
        self.regression = nn.Sequential(

            nn.Conv2d(
                128, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64, 1,
                kernel_size=1
            ),

            nn.ReLU()
        )

    def forward(self, x):

        x = self.features(x)

        x = self.regression(x)

        return x