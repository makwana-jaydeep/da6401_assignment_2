import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class OxfordIIITPetDataset(Dataset):
    """
    Oxford-IIIT Pet dataset for multi-task learning.
    Returns image, class label, bounding box [cx, cy, w, h] in pixel space, segmentation mask.
    """

    def __init__(self, root: str, split: str = "trainval", transform=None, mask_transform=None):
        """
        Args:
            root: Path to dataset root (contains 'images/', 'annotations/').
            split: 'trainval' or 'test'.
            transform: Albumentations transform applied to image + bbox + mask together.
            mask_transform: Additional mask-only transforms (rarely needed).
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        self.image_dir = os.path.join(root, "images")
        self.xml_dir = os.path.join(root, "annotations", "xmls")
        self.mask_dir = os.path.join(root, "annotations", "trimaps")

        split_file = os.path.join(root, "annotations", f"{split}.txt")
        self.samples = self._parse_split(split_file)

    def _parse_split(self, split_file):
        samples = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name = parts[0]
                class_id = int(parts[1]) - 1  # 0-indexed
                samples.append((name, class_id))
        return samples

    def _load_bbox(self, name):
        xml_path = os.path.join(self.xml_dir, f"{name}.xml")
        if not os.path.exists(xml_path):
            return None
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        obj = root.find("object")
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        bw = xmax - xmin
        bh = ymax - ymin
        return [cx, cy, bw, bh], w, h

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, class_id = self.samples[idx]

        img_path = os.path.join(self.image_dir, f"{name}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))

        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        mask = np.array(Image.open(mask_path))
        # Trimap: 1=foreground, 2=background, 3=border -> 0-indexed
        mask = (mask - 1).clip(0, 2).astype(np.uint8)

        bbox_data = self._load_bbox(name)
        if bbox_data is not None:
            bbox, orig_w, orig_h = bbox_data
        else:
            h, w = image.shape[:2]
            bbox = [w / 2.0, h / 2.0, float(w), float(h)]
            orig_w, orig_h = w, h

        if self.transform is not None:
            h, w = image.shape[:2]
            # Convert cx,cy,w,h -> x_min,y_min,w,h for albumentations
            bx = bbox[0] - bbox[2] / 2
            by = bbox[1] - bbox[3] / 2
            bw = bbox[2]
            bh = bbox[3]
            # Clip to valid range
            bx = max(0.0, min(bx, w - 1))
            by = max(0.0, min(by, h - 1))
            bw = min(bw, w - bx)
            bh = min(bh, h - by)

            transformed = self.transform(
                image=image,
                mask=mask,
                bboxes=[[bx, by, bw, bh]],
                bbox_labels=[0],
            )
            image = transformed["image"]
            mask = transformed["mask"]
            boxes = transformed["bboxes"]
            if len(boxes) > 0:
                bx2, by2, bw2, bh2 = boxes[0]
                bbox = [bx2 + bw2 / 2, by2 + bh2 / 2, bw2, bh2]
            # else keep original (rare edge case)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        label_tensor = torch.tensor(class_id, dtype=torch.long)

        return image, label_tensor, bbox_tensor, mask