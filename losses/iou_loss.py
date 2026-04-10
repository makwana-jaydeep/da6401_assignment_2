import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    1 - IoU loss for axis-aligned bounding boxes in [cx, cy, w, h] format.
    Boxes are converted to corner format internally before computing overlap.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        allowed = {"mean", "sum", "none"}
        if reduction not in allowed:
            raise ValueError(f"reduction must be one of {allowed}, got '{reduction}'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # Unpack centre-format into corner coordinates
        px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        gx1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        gy1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        gx2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        gy2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Overlap rectangle
        ix1 = torch.max(px1, gx1)
        iy1 = torch.max(py1, gy1)
        ix2 = torch.min(px2, gx2)
        iy2 = torch.min(py2, gy2)

        # Clamp prevents negative area for non-overlapping boxes
        overlap_w = (ix2 - ix1).clamp(min=0)
        overlap_h = (iy2 - iy1).clamp(min=0)
        intersection = overlap_w * overlap_h

        area_pred   = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        area_target = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
        union       = area_pred + area_target - intersection + self.eps

        iou  = intersection / union
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss