"""Training script for classification, localization, and segmentation tasks."""

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

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train=True):
    if train:
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.3),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"], min_visibility=0.3),
        )
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"], min_visibility=0.0),
    )


def dice_score(pred_mask, true_mask, num_classes=3, eps=1e-6):
    pred_mask = pred_mask.argmax(dim=1)
    scores = []
    for c in range(num_classes):
        p = (pred_mask == c).float()
        t = (true_mask == c).float()
        scores.append((2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps))
    return torch.stack(scores).mean().item()


def train_classifier(args):
    wandb.init(project=args.wandb_project, name="classifier", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", transform=get_transforms(True))
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_ds.dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss /= total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        wandb.log({"epoch": epoch + 1, "cls/train_loss": train_loss, "cls/train_acc": train_acc,
                   "cls/val_loss": val_loss, "cls/val_acc": val_acc})
        print(f"[Classifier] Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "classifier.pth"))

    wandb.finish()


def train_localizer(args):
    wandb.init(project=args.wandb_project, name="localizer", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", transform=get_transforms(True))
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_ds.dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    clf_ckpt = os.path.join(args.checkpoint_dir, "classifier.pth")
    if os.path.exists(clf_ckpt):
        from models.classification import VGG11Classifier
        clf = VGG11Classifier()
        clf.load_state_dict(torch.load(clf_ckpt, map_location="cpu"))
        model.encoder.load_state_dict(clf.encoder.state_dict())
        print("Loaded pretrained encoder from classifier checkpoint.")

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen.")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    best_val_loss = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        total = 0
        for images, _, bboxes, _ in train_loader:
            images, bboxes = images.to(device), bboxes.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = mse_loss(preds, bboxes) + iou_loss(preds, bboxes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            total += images.size(0)

        scheduler.step()
        train_loss /= total

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_total = 0
        with torch.no_grad():
            for images, _, bboxes, _ in val_loader:
                images, bboxes = images.to(device), bboxes.to(device)
                preds = model(images)
                LAMBDA_MSE = 0.01  # scale down MSE to match IoU's [0,1] range
                loss = LAMBDA_MSE * mse_loss(preds, bboxes) + iou_loss(preds, bboxes)
                val_loss += loss.item() * images.size(0)
                val_iou += (1 - iou_loss(preds, bboxes).item()) * images.size(0)
                val_total += images.size(0)
        val_loss /= val_total
        val_iou /= val_total

        wandb.log({"epoch": epoch + 1, "loc/train_loss": train_loss, "loc/val_loss": val_loss, "loc/val_iou": val_iou})
        print(f"[Localizer] Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "localizer.pth"))

    wandb.finish()


def train_unet(args):
    wandb.init(project=args.wandb_project, name=f"unet_{args.finetune_strategy}", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", transform=get_transforms(True))
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_ds.dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    clf_ckpt = os.path.join(args.checkpoint_dir, "classifier.pth")
    if os.path.exists(clf_ckpt):
        from models.classification import VGG11Classifier
        clf = VGG11Classifier()
        clf.load_state_dict(torch.load(clf_ckpt, map_location="cpu"))
        model.encoder.load_state_dict(clf.encoder.state_dict())
        print("Loaded pretrained encoder from classifier checkpoint.")

    strategy = args.finetune_strategy
    if strategy == "frozen":
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif strategy == "partial":
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.encoder.block5.parameters():
            param.requires_grad = True
        for param in model.encoder.block4.parameters():
            param.requires_grad = True
    # "full" -> all params trainable (default)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_dice = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        total = 0
        for images, _, _, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            total += images.size(0)

        scheduler.step()
        train_loss /= total

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_total = 0
        val_pixel_correct = 0
        val_pixel_total = 0
        with torch.no_grad():
            for images, _, _, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice_score(logits, masks) * images.size(0)
                preds = logits.argmax(1)
                val_pixel_correct += (preds == masks).sum().item()
                val_pixel_total += masks.numel()
                val_total += images.size(0)

        val_loss /= val_total
        val_dice /= val_total
        val_pixel_acc = val_pixel_correct / val_pixel_total

        wandb.log({"epoch": epoch + 1, "seg/train_loss": train_loss, "seg/val_loss": val_loss,
                   "seg/val_dice": val_dice, "seg/val_pixel_acc": val_pixel_acc})
        print(f"[UNet-{strategy}] Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} dice={val_dice:.4f} pixel_acc={val_pixel_acc:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "unet.pth"))

    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["classify", "localize", "segment"])
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--finetune_strategy", type=str, default="full", choices=["frozen", "partial", "full"])
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "classify":
        train_classifier(args)
    elif args.task == "localize":
        train_localizer(args)
    elif args.task == "segment":
        train_unet(args)