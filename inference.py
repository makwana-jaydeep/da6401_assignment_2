"""Load saved checkpoints and run multi-task inference on a single image."""

import argparse
import os

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import CLASS_NAMES

_IMG_SIZE = 224
_MEAN     = [0.485, 0.456, 0.406]
_STD      = [0.229, 0.224, 0.225]

TRIMAP_COLORS = {0: "foreground", 1: "background", 2: "border"}


def preprocess(image_path):
    arr = np.array(Image.open(image_path).convert("RGB"))
    pipeline = A.Compose([
        A.Resize(_IMG_SIZE, _IMG_SIZE),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
    return pipeline(image=arr)["image"].unsqueeze(0)


def run_inference(image_path, checkpoint_dir="checkpoints", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskPerceptionModel(
        classifier_path=os.path.join(checkpoint_dir, "classifier.pth"),
        localizer_path =os.path.join(checkpoint_dir, "localizer.pth"),
        unet_path      =os.path.join(checkpoint_dir, "unet.pth"),
    ).to(device)
    model.eval()

    tensor = preprocess(image_path).to(device)
    with torch.no_grad():
        results = model(tensor)

    cls_logits = results["classification"][0]
    cls_idx    = cls_logits.argmax().item()
    cls_label  = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)

    bbox     = results["localization"][0].cpu().numpy()
    seg_mask = results["segmentation"][0].argmax(0).cpu().numpy()

    print(f"Predicted breed : {cls_label} (index {cls_idx})")
    print(f"Bounding box    : {bbox}  [cx, cy, w, h]")
    print(f"Segmentation    : mask shape {seg_mask.shape}")

    return cls_label, bbox, seg_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",          type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    run_inference(args.image, args.checkpoint_dir)