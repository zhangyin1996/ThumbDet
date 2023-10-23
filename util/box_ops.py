# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):  # (x_c, y_c) 框的中心坐标. 变成左上角和右下角坐标   [ok]
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):  # 变成中心坐标、宽、高
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)  # 计算ox 面积 [ok]
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # lt: left top, 交集矩形框的左上角坐标      (600,6,2) 维度是如何得到的?
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # rb: right bottom, 交集矩形框的右下角坐标  (600,6,2)

    wh = (rb - lt).clamp(min=0)  # (x0-x1), (y0-y1) 求出交集矩形框的宽和高
    inter = wh[:, :, 0] * wh[:, :, 1]  # 两个交集的面积. [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check  矩形框，右下角坐标要大于左上角
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)  # ZY: return iou and union(并集面积)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # lt: left top, 交集矩形框的左上角坐标      (600,6,2) 维度是如何得到的?
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # rb: right bottom, 交集矩形框的右下角坐标  (600,6,2)

    wh = (rb - lt).clamp(min=0)  # (x0-x1), (y0-y1) 求出交集矩形框的宽和高
    area = wh[:, :, 0] * wh[:, :, 1]  # 两个交集的面积. (N,M)

    return iou - (area - union) / area  # iou loss ?


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
