import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """
    U-Net with a VGG-11 encoder.
    Upsampling uses transposed convolutions; skip connections concatenate
    encoder feature maps with decoder activations at matching resolutions.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder    = VGG11Encoder(in_channels=in_channels)
        self.bottleneck = _double_conv(512, 1024)

        self.up5  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = _double_conv(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64 + 128, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _double_conv(32 + 64, 32)

        self.dropout = CustomDropout(p=dropout_p)
        self.head    = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neck, skips = self.encoder(x, return_features=True)

        d = self.bottleneck(neck)

        d = self.up5(d);  d = torch.cat([d, skips["f5"]], dim=1);  d = self.dec5(d)
        d = self.up4(d);  d = torch.cat([d, skips["f4"]], dim=1);  d = self.dec4(d)
        d = self.up3(d);  d = torch.cat([d, skips["f3"]], dim=1);  d = self.dec3(d)
        d = self.up2(d);  d = torch.cat([d, skips["f2"]], dim=1);  d = self.dec2(d)
        d = self.up1(d);  d = torch.cat([d, skips["f1"]], dim=1);  d = self.dec1(d)

        d = self.dropout(d)
        return self.head(d)