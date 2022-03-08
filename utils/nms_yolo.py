"""
File: nms_yolo.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description:nms for yolo format
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

label_to_key = {'dog': 0, 'person': 1, 'cube': 2}
colors = ['green', 'blue', 'red']


def find_jaccard_overlap(set_1, set_2):

    intersection = find_intersection(set_1, set_2)

    # (xmax - xmin) * (ymax - ymin)
    # format (xmin, ymin, xmax, ymax)
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = (areas_set_1.unsqueeze(1) +
             areas_set_2.unsqueeze(1)).reshape(intersection.shape) - intersection
    return intersection / (union + 1e-6)


def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1),
                             set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1),
                             set_2[:, 2:].unsqueeze(0))

    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)

    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def box_dist(set_1, set_2):
    centers_1 = ((set_1[:, 2] + set_1[:, 0]) * 0.5, (set_1[:, 3] + set_1[:, 1]) * 0.5)
    centers_one = torch.cat((centers_1[0], centers_1[1]), 0)
    centers_one = torch.reshape(centers_one, (2, -1)).T

    centers_2 = ((set_2[:, 2] + set_2[:, 0]) * 0.5, (set_2[:, 3] + set_2[:, 1]) * 0.5)
    centers_two = torch.cat((centers_2[0], centers_2[1]), 0)
    centers_two = torch.reshape(centers_two, (2, -1)).T
    cdist = torch.cdist(centers_one, centers_two, p=2)

    ar_1 = (set_1[:, 2] - set_1[:, 0]) / (set_1[:, 3] - set_1[:, 1])
    ar_2 = (set_2[:, 2] - set_2[:, 0]) / (set_2[:, 3] - set_2[:, 1])
    ar_dist = torch.abs(ar_2 - ar_1)

    return cdist, ar_dist


def nms(preds: Tensor,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45):
    '''
    @preds: all the prediction in the format
            N x [label, confidence, xmin, ymin, xmax, ymax]
    '''

    # filter only the ones which are above confidence
    preds = preds[preds[:, 1] > confidence_threshold]

    # sort them in descending order
    preds = preds[preds[:, 1].argsort(descending=True)]

    classes = preds[:, 0].unique()
    preds_after_nms = []
    for cls in classes:
        preds_cls = preds[preds[:, 0] == cls]
        while preds_cls.shape[0]:
            curr_box = preds_cls[0].unsqueeze(0)
            other_box = preds_cls[1:]
            if len(other_box):
                distances, ar_dist = box_dist(curr_box[:, 2:], other_box[:, 2:])
                cond1 = distances > 13
                cond2 = ar_dist > 0.1

                ious = find_jaccard_overlap(curr_box[:, 2:], other_box[:, 2:]).squeeze()
                cond3 = ious < iou_threshold
                cond = cond1 | cond3
                # preds_cls = other_box[cond[0], :]
                preds_cls = other_box[cond3, :]
            else:
                preds_cls = other_box

            preds_after_nms.append(curr_box.squeeze())

    return torch.stack(preds_after_nms)


def get_box_predictions(filepath: str,
                        dbg: bool = False):

    img_path = filepath.replace('txt', 'png')
    img_pil = Image.open(img_path)
    draw = ImageDraw.Draw(img_pil)
    img_arr = np.array(img_pil)
    img_h, img_w = img_arr.shape[0:2]
    preds = []
    with open(filepath) as f:
        for line in f:
            line = [float(x) for x in line.strip().split()]
            [label, cx, cy, w, h, conf] = line
            w = img_w * w
            h = img_h * h
            cx = cx * img_w
            cy = cy * img_h

            xmin = round(cx - w * 0.5)
            ymin = round(cy - h * 0.5)
            xmax = round(cx + w * 0.5)
            ymax = round(cy + h * 0.5)
            if dbg:
                draw.text((xmin + 2, ymin + 2), '5', stroke_width=2)
                draw.rounded_rectangle((xmin, ymin, xmax, ymax), radius=5,
                                       outline=colors[0], width=2)

            preds.append([label, conf, xmin, ymin, xmax, ymax])

    preds = torch.as_tensor(preds)
    return img_pil, preds


def draw_bounding_box(img: Tensor,
                      boxes: Tensor):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rounded_rectangle(box[2:].tolist(), radius=5, width=2)


if __name__ == "__main__":
    for f_name in Path('./nms_test').glob('*.txt'):
        # f_name = './nms_test/sample-2021-11-18.2ad0e28635bc33138b38e04e6f50ce83.txt'
        img, preds = get_box_predictions(str(f_name), dbg=False)
        preds_after_nms = nms(preds)
        draw_bounding_box(img, preds_after_nms)
        img.save(f'./output/{str(f_name).replace("txt", "png")}')
        # img.show()
