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

class MCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # Column 1
        # Small receptive field
        # ====================================================

        self.column1 = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )


        # ====================================================
        # Column 2
        # Medium receptive field
        # ====================================================

        self.column2 = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU()
        )


        # ====================================================
        # Column 3
        # Large receptive field
        # ====================================================

        self.column3 = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=9,
                padding=4
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=9,
                padding=4
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU()
        )


        # ====================================================
        # Density regression
        # ====================================================

        self.regression = nn.Sequential(

            nn.Conv2d(
                192,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                1,
                kernel_size=1
            )
        )


    def forward(self, x):

        # ====================================================
        # Three scale-specific columns
        # ====================================================

        column1_output = self.column1(x)

        column2_output = self.column2(x)

        column3_output = self.column3(x)


        # ====================================================
        # Concatenate feature maps
        # ====================================================

        features = torch.cat(
            [
                column1_output,
                column2_output,
                column3_output
            ],
            dim=1
        )


        # ====================================================
        # Density regression
        # ====================================================

        density = self.regression(
            features
        )


        # ====================================================
        # Convert 128x128 feature map
        # to required 64x64 density map
        # ====================================================

        density = nn.functional.interpolate(
            density,
            size=(
                64,
                64
            ),
            mode="bilinear",
            align_corners=False
        )


        return density