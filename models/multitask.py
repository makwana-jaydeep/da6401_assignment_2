import os

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout, SigmoidBBox
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class MultiTaskPerceptionModel(nn.Module):
    """
    Shared VGG11 backbone with three task heads:
    classification, localization, and segmentation.

    A single forward() call returns all three outputs simultaneously, sharing
    the full encoder computation (backbone + skip features).

    Weight loading
    --------------
    Weights are transferred from the three individually trained checkpoints.
    The backbone is initialised from the classifier checkpoint (best source of
    discriminative features). The loc_head and seg decoder come from their
    respective checkpoints.

    Architecture notes
    ------------------
    * loc_head ends with SigmoidBBox (NOT ReLU). Both individual models had a
      ReLU bug here; this unified model is corrected.
    * The gdown download calls that existed in the original template have been
      removed. Checkpoints are read directly from local paths that are passed
      as constructor arguments, which is simpler and avoids network dependency
      at instantiation time.
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
        import gdown
        gdown.download(id="1fPd3gsn7CB-LoX621QLUoj6j1YQQksL7", output=classifier_path, quiet=False)
        gdown.download(id="12iBT78Ptvb__K-h1jlPJc0MenLHU-XOJ", output=localizer_path, quiet=False)
        gdown.download(id="1mSZbQLzeLpHtoAcPpQs_Ka1RM486NiiH", output=unet_path, quiet=False)

        # ── Shared backbone ───────────────────────────────────────────────
        self.backbone = VGG11Encoder(in_channels=in_channels)

        # ── Classification head ───────────────────────────────────────────
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

        # ── Localisation head ─────────────────────────────────────────────
        # FIXED: was nn.ReLU(inplace=True) as the last layer in both the
        # standalone VGG11Localizer and the original version of this file.
        # ReLU caused dying-neuron gradients and unbounded MSE loss.
        # SigmoidBBox maps output to (0, image_size) — stable and correct.
        self.loc_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            SigmoidBBox(scale=image_size),   # was: nn.ReLU(inplace=True) ← BUG FIXED
        )

        # ── Segmentation decoder ──────────────────────────────────────────
        self.bottleneck = double_conv(512, 1024)

        self.up5  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = double_conv(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(256 + 512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(128 + 256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(64 + 128, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = double_conv(32 + 64, 32)

        self.seg_dropout = CustomDropout(p=dropout_p)
        self.seg_head    = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ── Load pretrained weights ───────────────────────────────────────
        self._load_weights(classifier_path, localizer_path, unet_path)

    # ── Weight transfer ────────────────────────────────────────────────────

    def _load_weights(
        self,
        classifier_path: str,
        localizer_path: str,
        unet_path: str,
    ) -> None:
        device = torch.device("cpu")

        if os.path.exists(unet_path):
            unet = VGG11UNet()
            unet.load_state_dict(torch.load(unet_path, map_location=device))
            self.backbone.load_state_dict(unet.encoder.state_dict())  # backbone from unet
            self.bottleneck.load_state_dict(unet.bottleneck.state_dict())
            self.up5.load_state_dict(unet.up5.state_dict())
            self.dec5.load_state_dict(unet.dec5.state_dict())
            self.up4.load_state_dict(unet.up4.state_dict())
            self.dec4.load_state_dict(unet.dec4.state_dict())
            self.up3.load_state_dict(unet.up3.state_dict())
            self.dec3.load_state_dict(unet.dec3.state_dict())
            self.up2.load_state_dict(unet.up2.state_dict())
            self.dec2.load_state_dict(unet.dec2.state_dict())
            self.up1.load_state_dict(unet.up1.state_dict())
            self.dec1.load_state_dict(unet.dec1.state_dict())
            self.seg_head.load_state_dict(unet.head.state_dict())
            print(f"[MultiTask] Loaded U-Net weights (+ backbone) from {unet_path}")
        else:
            print(f"[MultiTask] WARNING: U-Net checkpoint not found at {unet_path}.")

        if os.path.exists(classifier_path):
            clf = VGG11Classifier()
            clf.load_state_dict(torch.load(classifier_path, map_location=device))
            self.cls_head.load_state_dict(clf.classifier.state_dict())  # cls_head only, not backbone
            print(f"[MultiTask] Loaded classifier head from {classifier_path}")
        else:
            print(f"[MultiTask] WARNING: classifier checkpoint not found at {classifier_path}.")

        if os.path.exists(localizer_path):
            loc = VGG11Localizer()
            loc.load_state_dict(torch.load(localizer_path, map_location=device))
            self.loc_head.load_state_dict(loc.regressor.state_dict())
            print(f"[MultiTask] Loaded localizer weights from {localizer_path}")
        else:
            print(f"[MultiTask] WARNING: localizer checkpoint not found at {localizer_path}.")

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Single forward pass over the shared backbone, branching into three heads.

        Returns
        -------
        dict with keys:
            "classification"  — (B, 37) logits
            "localization"    — (B, 4)  bbox coords [cx, cy, w, h] in (0, 224)
            "segmentation"    — (B, 3, H, W) per-pixel class logits
        """
        # Shared encoder — produces bottleneck + all five skip-connection maps
        bottleneck, features = self.backbone(x, return_features=True)

        # ── Classification + Localisation (share the same pooled feature) ──
        pooled = self.adaptive_pool(bottleneck)   # (B, 512, 7, 7)
        flat   = torch.flatten(pooled, 1)          # (B, 25088)
        cls_out = self.cls_head(flat)              # (B, 37)
        loc_out = self.loc_head(flat)              # (B, 4) ∈ (0, 224)

        # ── Segmentation decoder ───────────────────────────────────────────
        s = self.bottleneck(bottleneck)            # (B, 1024, 7,   7)

        s = self.up5(s)                            # (B,  512, 14,  14)
        s = torch.cat([s, features["f5"]], dim=1)  # (B, 1024, 14,  14)
        s = self.dec5(s)                           # (B,  512, 14,  14)

        s = self.up4(s)                            # (B,  256, 28,  28)
        s = torch.cat([s, features["f4"]], dim=1)  # (B,  768, 28,  28)
        s = self.dec4(s)                           # (B,  256, 28,  28)

        s = self.up3(s)                            # (B,  128, 56,  56)
        s = torch.cat([s, features["f3"]], dim=1)  # (B,  384, 56,  56)
        s = self.dec3(s)                           # (B,  128, 56,  56)

        s = self.up2(s)                            # (B,   64, 112, 112)
        s = torch.cat([s, features["f2"]], dim=1)  # (B,  192, 112, 112)
        s = self.dec2(s)                           # (B,   64, 112, 112)

        s = self.up1(s)                            # (B,   32, 224, 224)
        s = torch.cat([s, features["f1"]], dim=1)  # (B,   96, 224, 224)
        s = self.dec1(s)                           # (B,   32, 224, 224)

        s = self.seg_dropout(s)
        seg_out = self.seg_head(s)                 # (B,    3, 224, 224)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }