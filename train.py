"""Training routines for classification, localisation, and segmentation."""

import argparse
import os

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import wandb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

IMG_SIZE = 224
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transforms(train=True):
    base = [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ]
    if train:
        augments = [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.3),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
        return A.Compose(
            augments,
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"], min_visibility=0.3),
        )
    return A.Compose(
        base,
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"], min_visibility=0.0),
    )


def dice_score(pred_mask, true_mask, num_classes=3, eps=1e-6):
    predicted = pred_mask.argmax(dim=1)
    per_class = []
    for c in range(num_classes):
        p = (predicted == c).float()
        t = (true_mask  == c).float()
        numerator   = 2 * (p * t).sum() + eps
        denominator = p.sum() + t.sum() + eps
        per_class.append(numerator / denominator)
    return torch.stack(per_class).mean().item()


def _build_loaders(args, use_augment):
    full_ds  = OxfordIIITPetDataset(args.data_dir, split="trainval", transform=get_transforms(True))
    val_n    = int(0.15 * len(full_ds))
    train_n  = len(full_ds) - val_n
    train_ds, val_ds = random_split(
        full_ds, [train_n, val_n], generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", transform=get_transforms(False))

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
    )


def train_classifier(args):
    wandb.init(project=args.wandb_project, name="classifier", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = _build_loaders(args, use_augment=True)

    model     = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for imgs, labels, _, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct      += (logits.argmax(1) == labels).sum().item()
            seen         += imgs.size(0)
        scheduler.step()
        train_loss = running_loss / seen
        train_acc  = correct / seen

        model.eval()
        v_loss, v_correct, v_seen = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                v_loss    += criterion(logits, labels).item() * imgs.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_seen    += imgs.size(0)
        val_loss = v_loss / v_seen
        val_acc  = v_correct / v_seen

        wandb.log({"epoch": epoch + 1, "cls/train_loss": train_loss, "cls/train_acc": train_acc,
                   "cls/val_loss": val_loss, "cls/val_acc": val_acc})
        print(f"[Classifier] {epoch+1}/{args.epochs} | loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "classifier.pth"))

    wandb.finish()


def train_localizer(args):
    wandb.init(project=args.wandb_project, name="localizer", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = _build_loaders(args, use_augment=True)

    model     = VGG11Localizer(dropout_p=args.dropout_p).to(device)
    clf_ckpt  = os.path.join(args.checkpoint_dir, "classifier.pth")
    if os.path.exists(clf_ckpt):
        from models.classification import VGG11Classifier
        pretrained = VGG11Classifier()
        pretrained.load_state_dict(torch.load(clf_ckpt, map_location="cpu"))
        model.encoder.load_state_dict(pretrained.encoder.state_dict())
        print("Encoder initialised from classifier checkpoint.")

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder parameters frozen.")

    optimizer  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4
    )
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    mse_fn     = nn.MSELoss()
    iou_fn     = IoULoss(reduction="mean")
    best_loss  = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, seen = 0.0, 0
        for imgs, _, boxes, _ in train_loader:
            imgs, boxes = imgs.to(device), boxes.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = mse_fn(preds, boxes) + iou_fn(preds, boxes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
            seen       += imgs.size(0)
        scheduler.step()
        train_loss = epoch_loss / seen

        model.eval()
        v_loss, v_iou, v_seen = 0.0, 0.0, 0
        MSE_WEIGHT = 0.01
        with torch.no_grad():
            for imgs, _, boxes, _ in val_loader:
                imgs, boxes = imgs.to(device), boxes.to(device)
                preds   = model(imgs)
                loss    = MSE_WEIGHT * mse_fn(preds, boxes) + iou_fn(preds, boxes)
                v_loss += loss.item() * imgs.size(0)
                v_iou  += (1 - iou_fn(preds, boxes).item()) * imgs.size(0)
                v_seen += imgs.size(0)
        val_loss = v_loss / v_seen
        val_iou  = v_iou  / v_seen

        wandb.log({"epoch": epoch + 1, "loc/train_loss": train_loss,
                   "loc/val_loss": val_loss, "loc/val_iou": val_iou})
        print(f"[Localizer] {epoch+1}/{args.epochs} | train {train_loss:.4f} "
              f"| val {val_loss:.4f} iou {val_iou:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "localizer.pth"))

    wandb.finish()


def train_unet(args):
    strategy = args.finetune_strategy
    wandb.init(project=args.wandb_project, name=f"unet_{strategy}", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = _build_loaders(args, use_augment=True)

    model    = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)
    clf_ckpt = os.path.join(args.checkpoint_dir, "classifier.pth")
    if os.path.exists(clf_ckpt):
        from models.classification import VGG11Classifier
        pretrained = VGG11Classifier()
        pretrained.load_state_dict(torch.load(clf_ckpt, map_location="cpu"))
        model.encoder.load_state_dict(pretrained.encoder.state_dict())
        print("Encoder initialised from classifier checkpoint.")

    if strategy == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif strategy == "partial":
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.encoder.block5.parameters():
            p.requires_grad = True
        for p in model.encoder.block4.parameters():
            p.requires_grad = True
    # strategy == "full": all parameters trainable by default

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_dice = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, seen = 0.0, 0
        for imgs, _, _, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
            seen       += imgs.size(0)
        scheduler.step()
        train_loss = epoch_loss / seen

        model.eval()
        v_loss = v_dice = v_px_correct = v_px_total = v_seen = 0
        with torch.no_grad():
            for imgs, _, _, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits      = model(imgs)
                v_loss      += criterion(logits, masks).item() * imgs.size(0)
                v_dice      += dice_score(logits, masks) * imgs.size(0)
                preds        = logits.argmax(1)
                v_px_correct += (preds == masks).sum().item()
                v_px_total   += masks.numel()
                v_seen       += imgs.size(0)
        val_loss     = v_loss / v_seen
        val_dice     = v_dice / v_seen
        val_pixel_acc = v_px_correct / v_px_total

        wandb.log({"epoch": epoch + 1, "seg/train_loss": train_loss, "seg/val_loss": val_loss,
                   "seg/val_dice": val_dice, "seg/val_pixel_acc": val_pixel_acc})
        print(f"[UNet-{strategy}] {epoch+1}/{args.epochs} | loss {train_loss:.4f} "
              f"| val {val_loss:.4f} dice {val_dice:.4f} px_acc {val_pixel_acc:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "unet.pth"))

    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",             type=str,   required=True, choices=["classify", "localize", "segment"])
    p.add_argument("--data_dir",         type=str,   default="dataset")
    p.add_argument("--checkpoint_dir",   type=str,   default="checkpoints")
    p.add_argument("--epochs",           type=int,   default=20)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--dropout_p",        type=float, default=0.5)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--freeze_encoder",   action="store_true")
    p.add_argument("--finetune_strategy",type=str,   default="full", choices=["frozen", "partial", "full"])
    p.add_argument("--wandb_project",    type=str,   default="da6401-assignment2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dispatch = {
        "classify": train_classifier,
        "localize": train_localizer,
        "segment":  train_unet,
    }
    dispatch[args.task](args)