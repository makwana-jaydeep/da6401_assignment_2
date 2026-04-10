import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """VGG-11 encoder paired with a 3-layer FC head for 37-breed classification."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.3):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x, return_features=False)
        feat = self.adaptive_pool(feat)
        feat = torch.flatten(feat, start_dim=1)
        return self.classifier(feat)