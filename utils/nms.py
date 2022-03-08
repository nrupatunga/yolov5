"""
File: nms.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: non maximum supression for yolov1
"""
import base64
import io
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision import transforms

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


def nms(preds: Tensor,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5):
    '''
    @preds: all the prediction in the format
            N x [label, confidence, xmin, ymin, xmax, ymax]
    '''

    # filter only the ones which are above confidence
    preds = preds[preds[:, 1] > confidence_threshold]
    __import__('pdb').set_trace()

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
                ious = find_jaccard_overlap(curr_box[:, 2:], other_box[:, 2:]).squeeze()
                preds_cls = other_box[ious < iou_threshold, :]
            else:
                preds_cls = other_box

            preds_after_nms.append(curr_box.squeeze())

    return torch.stack(preds_after_nms)


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_pil = Image.open(f)
    img_arr = np.array(img_pil)
    return img_arr, img_pil


def get_box_predictions(filepath: str,
                        dbg: bool = True):

    with open(filepath) as f:
        data = json.load(f)
        img, img_pil = img_b64_to_arr(data['imageData'])
        img = transforms.ToTensor()(img).unsqueeze(0)
        draw = ImageDraw.Draw(img_pil)

        shapes = data['shapes']
        preds = []
        for shape in shapes:
            # labels
            label, confidence = shape['label'].split('_')
            label = label_to_key[label]
            confidence = float(confidence)

            # bounding box
            ((xmin, ymin), (xmax, ymax)) = shape['points']

            # drawing
            if dbg:
                draw.text((xmin + 2, ymin + 2), shape['label'], stroke_width=2)
                draw.rounded_rectangle((xmin, ymin, xmax, ymax), radius=5,
                                       outline=colors[label], width=2)

            preds.append([label, confidence, xmin, ymin, xmax, ymax])

        preds = torch.as_tensor(preds)
        return img_pil, preds


def draw_bounding_box(img: Tensor,
                      boxes: Tensor):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rounded_rectangle(box[2:].tolist(), radius=5, width=4)


if __name__ == "__main__":
    img, preds = get_box_predictions('./nms_test/person_dog.json', dbg=True)
    preds_after_nms = nms(preds)
    draw_bounding_box(img, preds_after_nms)
    img.show()
