import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11 encoder + regression head outputting [x_center, y_center, w, h] in pixel space."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, return_features=False)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)