import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout, SigmoidBBox


class VGG11Localizer(nn.Module):
    """
    VGG11 encoder + regression head outputting [x_center, y_center, w, h]
    in pixel space (values in [0, 224] after SigmoidBBox activation).

    Design choices
    --------------
    * The regression head mirrors the VGG11 classifier head (4096 → 1024 → 4)
      but ends with SigmoidBBox instead of ReLU. ReLU was removed because:
        - It causes dying neurons: any pre-activation value ≤ 0 produces zero
          gradient and the neuron never recovers.
        - It is unbounded, so early-training random weights can produce
          arbitrarily large predictions and explode the MSE loss.
      SigmoidBBox keeps all four outputs in (0, scale), gradients are always
      nonzero, and the output range matches the target coordinate space exactly.

    * CustomDropout p=0.5 is applied after each hidden FC layer to regularise
      the large 4096-dim layers. It is NOT applied to the 4-dim output since
      randomly zeroing bounding box coordinates would corrupt the regression
      signal completely.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5,
                 image_size: float = 224.0):
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
            SigmoidBBox(scale=image_size),   # was: nn.ReLU(inplace=True) ← BUG FIXED
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, return_features=False)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)