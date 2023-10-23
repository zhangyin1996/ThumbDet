# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

import matplotlib.pyplot as plt
import numpy as np
import os
from detr_demo import draw_bboxes
DOWN_SAMPLE_EPOCH = 5    # added

def train_one_epoch(model: torch.nn.Module, model_t: torch.nn.Module,criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, down_sample_epoch: int = 5):
    model.train()

    # 0414
    model_t.eval()

    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    down_sample_dir = os.path.join(os.getcwd(), 'exps', 'downsample', header)   # added
    if not os.path.exists(down_sample_dir):
        os.mkdir(down_sample_dir)

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        # -------------------------------------------------------------------------------------
        # Debug
        # outputs = model(samples, 5, is_down_sample=True)  # [debug]
        # with torch.no_grad():  # 0428 add
        #     outputs_t = model_t(samples, 49, is_down_sample=False)  # 49 epoch
        # outputs['pred_logits_t'] = outputs_t['pred_logits']
        # outputs['srcs_base'] = outputs_t['srcs']  # zy add 2022-05-30
        #
        # loss_dict = criterion(outputs, targets, 5)  # [debug]
        # weight_dict = criterion.weight_dict
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # ---------------------------------------------------------------------------------------
        outputs = model(samples, epoch, is_down_sample=True)  # goto deformable_detr.py forward   add epoch, is_down_sample=True
        if epoch >= down_sample_epoch:
            with torch.no_grad():
                outputs_t = model_t(samples, 49, is_down_sample=False)
            outputs['pred_logits_t'] = outputs_t['pred_logits']
            outputs['srcs_base'] = outputs_t['srcs']

        loss_dict = criterion(outputs, targets, epoch)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

# ZY: add image show+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if _ % 2000 == 0:
        #     for i, img in enumerate(outputs['downsample_input']):
        #         # [N, C, H, W] -> [C, H, W]
        #         im = np.squeeze(img.detach().cpu().numpy())
        #         # [C, H, W] -> [H, W, C]
        #         im = np.transpose(im, [1, 2, 0])
        #         im = np.clip(im, a_min=0.0, a_max=1.0)  # ZY: Floating point image RGB values must be in the 0..1 range.
        #
        #         filename = os.path.join(down_sample_dir, header + '_' + 'loader:'+ str(_) + '_' + 'bsID:'+ str(i) + '.jpg')
        #
        #         plt.imsave(filename, im)
# Draw BBox================================================================================================================
            # draw_bboxes(outputs['downsample_input'], outputs, header, _)
# Draw BBox================================================================================================================

        if _ % print_freq == 0:
            print('=====================loss_downsample=%.10e' %(loss_dict['loss_downsample']))
            if epoch >= down_sample_epoch:
                print('=====================loss_kd=%.10e' % (loss_dict['loss_kd']))
                print('=====================loss_fm=%.10e' % (loss_dict['loss_fm']))

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, model_t, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    # 0414
    # model_t = model
    model_t.eval()
    # checkpoint = torch.load('./r50_deformable_detr-checkpoint.pth', map_location='cpu')
    # model_t.load_state_dict(checkpoint['model'], strict=False)

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():  # not meet
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    count = -1  # add
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, DOWN_SAMPLE_EPOCH, is_down_sample=True)  # zy add DOWN_SAMPLE_EPOCH = 5
        outputs_t = model_t(samples, 49, is_down_sample=False) # 49 epoch
        outputs['pred_logits_t'] = outputs_t['pred_logits']
        outputs['srcs_base'] = outputs_t['srcs']

        loss_dict = criterion(outputs, targets, DOWN_SAMPLE_EPOCH) # zy add 5
        weight_dict = criterion.weight_dict
# Draw BBox================================================================================================================
#         count += 1  # zy add
#         draw_bboxes(samples.tensors, outputs, header, count)
# Draw BBox================================================================================================================

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
