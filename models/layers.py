import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Inverted dropout implemented without torch.nn.Dropout."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = torch.empty_like(x).bernoulli_(1.0 - self.p)
        return x * mask / (1.0 - self.p)


class SigmoidBBox(nn.Module):
    """
    Final activation for bounding box regression heads.

    Replaces the naive ReLU that was previously used. ReLU causes two problems:
      1. Dead neurons: once a unit outputs 0, the gradient is also 0 and the
         weight never updates ('dying ReLU' problem).
      2. Unbounded output: ReLU allows arbitrarily large predictions, making
         MSE loss explode early in training when weights are random.

    Sigmoid maps logits to (0, 1) and multiplying by `scale` (= image size in
    pixels, default 224) maps predictions to (0, 224). This matches the dataset's
    [cx, cy, w, h] coordinate space exactly, keeps gradients alive everywhere,
    and bounds the output so MSE stays numerically stable from epoch 1.
    """

    def __init__(self, scale: float = 224.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale