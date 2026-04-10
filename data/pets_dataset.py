import os
import random
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
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_NAMES)}


class OxfordIIITPetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform=None,
        mask_transform=None,
        require_bbox: bool = False,
    ):
        self.root           = root
        self.split          = split
        self.transform      = transform
        self.mask_transform = mask_transform
        self.require_bbox   = require_bbox

        self.image_dir = os.path.join(root, "images")
        self.xml_dir   = os.path.join(root, "annotations", "xmls")
        self.mask_dir  = os.path.join(root, "annotations", "trimaps")

        self.samples = self._parse_split()

    def _parse_split(self):
        list_file = os.path.join(self.root, "annotations", "list.txt")
        entries = []

        with open(list_file) as fh:
            for raw in fh:
                row = raw.strip()
                if not row or row.startswith("#"):
                    continue
                cols     = row.split()
                stem     = cols[0]
                cls_id   = int(cols[1]) - 1
                img_path  = os.path.join(self.image_dir, f"{stem}.jpg")
                msk_path  = os.path.join(self.mask_dir,  f"{stem}.png")
                xml_path  = os.path.join(self.xml_dir,   f"{stem}.xml")

                if not os.path.exists(img_path) or not os.path.exists(msk_path):
                    continue
                if self.require_bbox and not os.path.exists(xml_path):
                    continue

                entries.append((stem, cls_id))

        # Deterministic shuffle then split
        entries_sorted = sorted(entries)
        rng = random.Random(42)
        rng.shuffle(entries_sorted)
        cutoff = int(0.85 * len(entries_sorted))

        return entries_sorted[:cutoff] if self.split == "trainval" else entries_sorted[cutoff:]

    def _load_bbox(self, stem):
        xml_path = os.path.join(self.xml_dir, f"{stem}.xml")
        if not os.path.exists(xml_path):
            return None
        root_node = ET.parse(xml_path).getroot()
        sz   = root_node.find("size")
        orig_w = int(sz.find("width").text)
        orig_h = int(sz.find("height").text)
        bndbox = root_node.find("object").find("bndbox")
        x0 = float(bndbox.find("xmin").text)
        y0 = float(bndbox.find("ymin").text)
        x1 = float(bndbox.find("xmax").text)
        y1 = float(bndbox.find("ymax").text)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        bw = x1 - x0
        bh = y1 - y0
        return [cx, cy, bw, bh], orig_w, orig_h

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, cls_id = self.samples[idx]

        img = np.array(Image.open(os.path.join(self.image_dir, f"{stem}.jpg")).convert("RGB"))
        msk = np.array(Image.open(os.path.join(self.mask_dir,  f"{stem}.png")))
        msk = (msk - 1).clip(0, 2).astype(np.uint8)

        bbox_info = self._load_bbox(stem)
        if bbox_info is not None:
            bbox, _ow, _oh = bbox_info
        else:
            h, w = img.shape[:2]
            bbox = [w / 2.0, h / 2.0, float(w), float(h)]

        if self.transform is not None:
            h, w    = img.shape[:2]
            bx      = max(0.0, bbox[0] - bbox[2] / 2)
            by      = max(0.0, bbox[1] - bbox[3] / 2)
            bw_clip = min(bbox[2], w - bx)
            bh_clip = min(bbox[3], h - by)

            out   = self.transform(
                image=img, mask=msk,
                bboxes=[[bx, by, bw_clip, bh_clip]],
                bbox_labels=[0],
            )
            img = out["image"]
            msk = out["mask"]
            remaining = out["bboxes"]
            if remaining:
                rx, ry, rw, rh = remaining[0]
                bbox = [rx + rw / 2, ry + rh / 2, rw, rh]
            else:
                bbox = [112.0, 112.0, 224.0, 224.0]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            msk = torch.from_numpy(msk).long()

        return img, torch.tensor(cls_id, dtype=torch.long), torch.tensor(bbox, dtype=torch.float32), msk