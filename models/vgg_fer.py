# models/vgg_fer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGFER(nn.Module):
    """
    VGG-style CNN for FER2013 (grayscale, 7 classes).

    - 4 conv blocks, each: 2 x (Conv3x3 + BatchNorm + ReLU) + MaxPool2d(2)
    - 3 fully-connected layers.
    - Uses AdaptiveAvgPool2d so it works with 40x40 or 48x48 inputs.
    """

    def __init__(self, num_classes: int = 7, dropout_p: float = 0.5):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        # 1 x 48 x 48 (or 40 x 40) input
        self.features = nn.Sequential(
            conv_block(1, 64),    # -> 64 x 24 x 24 (or 20 x 20)
            conv_block(64, 128),  # -> 128 x 12 x 12 (or 10 x 10)
            conv_block(128, 256), # -> 256 x 6 x 6 (or 5 x 5)
            conv_block(256, 512), # -> 512 x 3 x 3 (or 2 x 2)
        )

        # make output spatial size fixed at 3x3 regardless of input
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)           # (B, 512, H', W')
        x = self.avgpool(x)            # (B, 512, 3, 3)
        x = x.view(x.size(0), -1)      # (B, 512*3*3)
        x = self.classifier(x)         # (B, num_classes)
        return x


if __name__ == "__main__":
    model = VGGFER(num_classes=7)
    x = torch.randn(4, 1, 48, 48)
    y = model(x)
    print(y.shape)  # (4, 7)
