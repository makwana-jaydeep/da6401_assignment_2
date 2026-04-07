import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss

class IoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        pred_x1 = pred_boxes[:, 0] - (pred_boxes[:, 2] / 2.0)
        pred_y1 = pred_boxes[:, 1] - (pred_boxes[:, 3] / 2.0)
        pred_x2 = pred_boxes[:, 0] + (pred_boxes[:, 2] / 2.0)
        pred_y2 = pred_boxes[:, 1] + (pred_boxes[:, 3] / 2.0)
        preds_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

        tgt_x1 = target_boxes[:, 0] - (target_boxes[:, 2] / 2.0)
        tgt_y1 = target_boxes[:, 1] - (target_boxes[:, 3] / 2.0)
        tgt_x2 = target_boxes[:, 0] + (target_boxes[:, 2] / 2.0)
        tgt_y2 = target_boxes[:, 1] + (target_boxes[:, 3] / 2.0)
        tgts_xyxy = torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=1)

        # GIoU resolves the zero gradient issue for non-overlapping boxes
        loss = generalized_box_iou_loss(preds_xyxy, tgts_xyxy, reduction=self.reduction)
        return loss