# DA6401 — Assignment 2: Visual Perception Pipeline

**Name:** Jaydeep Makwana

**Roll Number:** DA25M013

---

##  Weights & Biases Report

[**View Full W&B Report →**](https://wandb.ai/da25m013-indian-institute-of-technology-madras/da6401-as2/reports/DA6401-Assignment-2-Visual-Perception-Pipeline--VmlldzoxNjQ2Nzc4NA?accessToken=2ow5cu6k30g5ln2ugqfaogwottr8s9rv6djmd7ju4ne3msg2v75lx7df5a8d2ktz)

---

## Overview

This assignment builds a complete multi-task visual perception pipeline on the **Oxford-IIIT Pet Dataset**, covering three tasks:

| Task | Description | Model |
|------|-------------|-------|
| Classification | 37-breed pet identification | VGG11 + Custom Dropout |
| Localization | Bounding box regression | VGG11 Encoder + Regression Head |
| Segmentation | Pixel-wise trimap prediction | VGG11 U-Net |

All three tasks share a single VGG11 backbone in the final unified `MultiTaskPerceptionModel`.

---

## Project Structure

```
da6401_assignment_2/
├── data/
│   └── pets_dataset.py          # Oxford-IIIT Pet Dataset loader
├── losses/
│   ├── __init__.py
│   └── iou_loss.py              # Custom IoU loss (nn.Module)
├── models/
│   ├── __init__.py
│   ├── layers.py                # Custom Dropout (nn.Module)
│   ├── vgg11.py                 # VGG11 encoder from scratch
│   ├── classification.py        # VGG11 classifier head
│   ├── localization.py          # VGG11 localizer head
│   ├── segmentation.py          # VGG11 U-Net
│   └── multitask.py             # Unified multi-task model
├── checkpoints/
│   └── checkpoints.md
├── train.py
├── inference.py
└── requirements.txt
```

---

## Implementation Details

### Task 1 — Classification

- VGG11 built from scratch using `nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d`
- **Custom Dropout** implemented by subclassing `nn.Module` with inverted dropout scaling (`x / (1 - p)` at train time, identity at eval)
- BatchNorm placed after every convolutional layer for training stability
- Dropout placed in the fully-connected head at `p=0.5`
- Loss: `CrossEntropyLoss`

### Task 2 — Localization

- VGG11 convolutional backbone used as encoder (fine-tuned, not frozen)
- Regression head outputs `[cx, cy, w, h]` in pixel space via `Sigmoid × 224`
- **Custom IoU Loss** implemented by subclassing `nn.Module`
- Training loss: `MSE(normalised) + IoU`
- Only samples with XML bounding box annotations used for training

### Task 3 — Segmentation

- U-Net style decoder with skip connections from all 5 VGG11 encoder blocks
- Upsampling via `nn.ConvTranspose2d` (no bilinear interpolation)
- Feature fusion via channel-wise concatenation at each decoder stage
- Loss: `CrossEntropyLoss` over 3 classes (background, foreground, border)
- Full fine-tuning strategy yielded best Dice score

### Task 4 — Unified Multi-Task Model

- Single shared VGG11 backbone
- Three task heads branch from the shared bottleneck
- Single `forward()` pass returns classification logits, bounding box, and segmentation mask simultaneously
- Pretrained weights loaded from individual task checkpoints

---

## Results

| Task | Metric | Score |
|------|--------|-------|
| Classification | Macro F1 | 0.667 |
| Localization | Acc @ IoU ≥ 0.50 | 20.0% |
| Localization | Acc @ IoU ≥ 0.75 | 0.0% |
| Segmentation | Mean Dice | 0.86 |
| Segmentation | Pixel Accuracy | 0.92 |

---

## Setup & Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/da6401_assignment_2.git
cd da6401_assignment_2

# Install dependencies
pip install torch torchvision albumentations wandb gdown scikit-learn
```

### Dataset

Download the Oxford-IIIT Pet Dataset:

```bash
# Images
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# Annotations
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

tar -xf images.tar.gz
tar -xf annotations.tar.gz
```

### Pretrained Checkpoints

Checkpoints are hosted on Google Drive and auto-downloaded by `multitask.py` via `gdown`.

| File | Google Drive ID |
|------|----------------|
| `classifier.pth` | `1fPd3gsn7CB-LoX621QLUoj6j1YQQksL7` |
| `localizer.pth` | `12iBT78Ptvb__K-h1jlPJc0MenLHU-XOJ` |
| `unet.pth` | `1mSZbQLzeLpHtoAcPpQs_Ka1RM486NiiH` |

---

## Training

```python
# Example: train classifier
from models.classification import VGG11Classifier
import torch.nn as nn

model = VGG11Classifier(num_classes=37)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

Key training configuration:

| Setting | Value |
|---------|-------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Learning rate | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=3) |
| Normalisation | ImageNet mean/std |
| Train / Val split | 85% / 15% (seed=42) |

---

## Inference

```python
from models.multitask import MultiTaskPerceptionModel
import torch

model = MultiTaskPerceptionModel()
model.eval()

# Single forward pass → all three outputs
with torch.no_grad():
    output = model(image_tensor)

print(output["classification"])  # (B, 37) logits
print(output["localization"])    # (B, 4)  [cx, cy, w, h]
print(output["segmentation"])    # (B, 3, H, W) logits
```

---

## W&B Experiments

All experiments tracked at:
[https://wandb.ai/da25m013-indian-institute-of-technology-madras/da6401-as2](https://wandb.ai/da25m013-indian-institute-of-technology-madras/da6401-as2)

| Experiment Group | Description |
|-----------------|-------------|
| `2.1_batchnorm_effect` | With vs without BatchNorm — activation statistics |
| `2.2_dropout_comparison` | Dropout p=0.0 / 0.2 / 0.5 ablation |
| `2.3_transfer_learning` | Frozen / partial / full fine-tuning strategies |
| `2.4_feature_maps` | First and last conv layer visualisations |
| `2.5_detection` | Bounding box predictions with IoU per sample |
| `2.6_segmentation_curves` | Dice and pixel accuracy per epoch |
| `2.7_novel_images` | In-the-wild pipeline showcase |
| `2.8_training_curves` | Full training history for all three tasks |

---

## Dependencies

```
torch>=2.0
torchvision>=0.15
albumentations>=2.0.0
wandb
gdown
scikit-learn
numpy
Pillow
```
