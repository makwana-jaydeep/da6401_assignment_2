import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout, SigmoidBBox


class VGG11Localizer(nn.Module):
    """VGG-11 encoder with a regression head predicting [cx, cy, w, h]."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder      = VGG11Encoder(in_channels=in_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.image_size   = 224
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x, return_features=False)
        feat = self.adaptive_pool(feat)
        feat = torch.flatten(feat, start_dim=1)
        return self.regressor(feat) * self.image_size