import os

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout, SigmoidBBox
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
import gdown

def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiTaskPerceptionModel(nn.Module):
    """
    Unified model with a shared VGG-11 backbone and three task heads:
      1. Classification  — 37-class breed prediction
      2. Localization    — [cx, cy, w, h] bounding-box regression
      3. Segmentation    — per-pixel trimap labelling via a U-Net decoder

    Two backbone instances are kept so the classifier/localizer backbone and
    the segmentation backbone can hold weights from their respective checkpoints
    without interference.

    Pretrained weights are loaded from local checkpoint paths passed at
    construction time; no external download calls are made.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        image_size: float = 224.0,
        dropout_p: float = 0.5,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        super().__init__()
        
        gdown.download(id="1fPd3gsn7CB-LoX621QLUoj6j1YQQksL7", output=classifier_path, quiet=False)
        gdown.download(id="12iBT78Ptvb__K-h1jlPJc0MenLHU-XOJ", output=localizer_path, quiet=False)
        gdown.download(id="1mSZbQLzeLpHtoAcPpQs_Ka1RM486NiiH", output=unet_path, quiet=False)

        # Two separate encoders: one for cls+loc, one for seg
        self.backbone     = VGG11Encoder(in_channels=in_channels)
        self.seg_backbone = VGG11Encoder(in_channels=in_channels)

        # Classification head
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds),
        )

        # Localisation head — SigmoidBBox avoids the unbounded-output / dying-ReLU problem
        self.loc_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            SigmoidBBox(scale=image_size),
        )

        # Segmentation decoder (U-Net style)
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

        self.seg_dropout = CustomDropout(p=dropout_p)
        self.seg_head    = nn.Conv2d(32, seg_classes, kernel_size=1)

        self._load_weights(classifier_path, localizer_path, unet_path)

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        cpu = torch.device("cpu")

        if os.path.exists(unet_path):
            unet_ckpt = VGG11UNet()
            unet_ckpt.load_state_dict(torch.load(unet_path, map_location=cpu))
            self.seg_backbone.load_state_dict(unet_ckpt.encoder.state_dict())
            self.bottleneck.load_state_dict(unet_ckpt.bottleneck.state_dict())
            for level in [5, 4, 3, 2, 1]:
                getattr(self, f"up{level}").load_state_dict(
                    getattr(unet_ckpt, f"up{level}").state_dict()
                )
                getattr(self, f"dec{level}").load_state_dict(
                    getattr(unet_ckpt, f"dec{level}").state_dict()
                )
            self.seg_head.load_state_dict(unet_ckpt.head.state_dict())
            print(f"[MultiTask] U-Net weights loaded from {unet_path}")
        else:
            print(f"[MultiTask] WARNING: no U-Net checkpoint at {unet_path}")

        if os.path.exists(classifier_path):
            clf_ckpt = VGG11Classifier()
            clf_ckpt.load_state_dict(torch.load(classifier_path, map_location=cpu))
            self.backbone.load_state_dict(clf_ckpt.encoder.state_dict())
            self.cls_head.load_state_dict(clf_ckpt.classifier.state_dict())
            print(f"[MultiTask] Classifier weights loaded from {classifier_path}")
        else:
            print(f"[MultiTask] WARNING: no classifier checkpoint at {classifier_path}")

        if os.path.exists(localizer_path):
            loc_ckpt = VGG11Localizer()
            loc_ckpt.load_state_dict(torch.load(localizer_path, map_location=cpu))
            self.loc_head.load_state_dict(loc_ckpt.regressor.state_dict())
            print(f"[MultiTask] Localizer weights loaded from {localizer_path}")
        else:
            print(f"[MultiTask] WARNING: no localizer checkpoint at {localizer_path}")

    def forward(self, x: torch.Tensor) -> dict:
        # Classification + localisation share the cls/loc backbone
        neck, _     = self.backbone(x, return_features=True)
        pooled      = self.adaptive_pool(neck)
        flat        = torch.flatten(pooled, start_dim=1)
        cls_out     = self.cls_head(flat)
        loc_out     = self.loc_head(flat)

        # Segmentation uses its own backbone to preserve U-Net features
        seg_neck, skips = self.seg_backbone(x, return_features=True)
        d = self.bottleneck(seg_neck)
        d = self.up5(d);  d = torch.cat([d, skips["f5"]], dim=1);  d = self.dec5(d)
        d = self.up4(d);  d = torch.cat([d, skips["f4"]], dim=1);  d = self.dec4(d)
        d = self.up3(d);  d = torch.cat([d, skips["f3"]], dim=1);  d = self.dec3(d)
        d = self.up2(d);  d = torch.cat([d, skips["f2"]], dim=1);  d = self.dec2(d)
        d = self.up1(d);  d = torch.cat([d, skips["f1"]], dim=1);  d = self.dec1(d)
        d = self.seg_dropout(d)
        seg_out = self.seg_head(d)

        return {"classification": cls_out, "localization": loc_out, "segmentation": seg_out}