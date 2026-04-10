import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Manual inverted dropout — does NOT use nn.Dropout internally."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Drop probability must be in [0, 1), received {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.p
        bernoulli_mask = torch.empty_like(x).bernoulli_(keep_prob)
        return x * bernoulli_mask / keep_prob


class SigmoidBBox(nn.Module):
    """
    Output activation for bounding-box regression.

    Why sigmoid instead of ReLU:
      - ReLU zeroes out negative activations and kills gradients for those
        neurons permanently (the 'dying ReLU' issue).
      - Unclamped ReLU lets predictions grow without bound, causing MSE loss
        to blow up in early epochs when weights are still random.
      Sigmoid squashes logits to (0, 1); scaling by `scale` (image width/height)
      brings predictions into the [0, scale] pixel range, which aligns with
      the [cx, cy, w, h] target format used by the dataset loader.
    """

    def __init__(self, scale: float = 224.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale