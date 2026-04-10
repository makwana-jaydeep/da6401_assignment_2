from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    """
    Convolutional backbone following the VGG-11 architecture
    (Simonyan & Zisserman, 2014), augmented with BatchNorm after each conv.
    Intermediate spatial feature maps can be returned for skip connections.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def _conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.block1 = nn.Sequential(_conv_block(in_channels, 64))
        self.pool1  = nn.MaxPool2d(2, 2)

        self.block2 = nn.Sequential(_conv_block(64, 128))
        self.pool2  = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            _conv_block(128, 256),
            _conv_block(256, 256),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.block4 = nn.Sequential(
            _conv_block(256, 512),
            _conv_block(512, 512),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.block5 = nn.Sequential(
            _conv_block(512, 512),
            _conv_block(512, 512),
        )
        self.pool5 = nn.MaxPool2d(2, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        s1 = self.block1(x);        x = self.pool1(s1)
        s2 = self.block2(x);        x = self.pool2(s2)
        s3 = self.block3(x);        x = self.pool3(s3)
        s4 = self.block4(x);        x = self.pool4(s4)
        s5 = self.block5(x);        neck = self.pool5(s5)

        if return_features:
            skips = {"f1": s1, "f2": s2, "f3": s3, "f4": s4, "f5": s5}
            return neck, skips

        return neck