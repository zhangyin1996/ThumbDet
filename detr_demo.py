import torch
import matplotlib.pyplot as plt
import os
import numpy as np

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.18], [0.301, 0.745, 0.933]]

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_h, img_w, c = size
    b = box_cxcywh_to_xyxy(out_bbox).cpu()
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(ndarray_img, prob, boxes, filename):
    # plt.figure(figsize=(16,10))
    plt.figure()
    plt.imshow(ndarray_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        fill=False, color=c, linewidth=2))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.savefig(filename)
    # plt.axis('off')
    # plt.show()
    # plt.imsave(filename, ndarray_img)

def draw_bboxes(img_tensor, outputs, header, count):
    """
    img_tensor: (B, C, H, W)
    outputs: outputs['pred_logits'] and outputs['pred_boxes']
    header: used to pic name head. 'Train' or 'Test'
    count: used to pic name
    """
    if header == 'Test:':
        draw_bbox_results_dir = os.path.join(os.getcwd(), 'exps', 'draw_bbox_results', 'test')
    else:
        draw_bbox_results_dir = os.path.join(os.getcwd(), 'exps', 'draw_bbox_results', 'train')
    if not os.path.exists(draw_bbox_results_dir):
        os.mkdir(draw_bbox_results_dir)

    if count % 50 == 0:
        for i, img in enumerate(img_tensor):
            filename = os.path.join(draw_bbox_results_dir,
                                    header + 'number_' + str(count) + '_' + 'bsID:' + str(i) + '.jpg')
            ##### 1.显示原始图
            # [N, C, H, W] -> [C, H, W]
            im = np.squeeze(img.detach().cpu().numpy())
            # [C, H, W] -> [H, W, C]
            im = np.transpose(im, [1, 2, 0])
            im = np.clip(im, a_min=0.0, a_max=1.0)  # ZY: Floating point image RGB values must be in the 0..1 range.
            ##### 2.开始画框

            pred_logits = outputs['pred_logits'][i]  # 每幅图的分类 (b,300,91) -> (300,91)
            prob = pred_logits.softmax(-1)
            keep = prob.max(-1).values > 0.7


            pred_boxes = outputs['pred_boxes'][i]  # 每幅图的 boxes

            bboxes_scaled = rescale_bboxes(pred_boxes[keep], im.shape)
            scores = prob[keep]
            # print(scores.shape)
            # print(bboxes_scaled.shape)
            plot_results(im, scores, bboxes_scaled, filename)

