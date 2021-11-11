import cv2
import numpy as np
import yaml

from utils.datasets import create_dataloader
from utils.general import xywh2xyxy
from utils.plots import Annotator, colors

imgsz = 256
batch_size = 1
gs = 32
hyp = './data/hyps/hyp.scratch.yaml'
with open(hyp, errors='ignore') as f:
    hyp = yaml.safe_load(f)  # load hyps dict

root_dir = '/home/nthere/2021/yolov5/data/simreal'
train_loader, dataset = create_dataloader(root_dir,
                                          imgsz,
                                          batch_size,
                                          gs,
                                          hyp=hyp,
                                          augment=True,
                                          cache=False)
pbar = enumerate(train_loader)

names = ['whole', 'half', 'third', 'fourth', 'sixth', 'eighth']

for i, (imgs, targets, paths, _) in pbar:
    labels = targets[:, 1]
    boxes = targets[:, 2:6]
    annotator = Annotator(np.ascontiguousarray(np.transpose(imgs[0].numpy(), (1, 2,
                                                                              0))))
    boxes = boxes * 256
    boxes = xywh2xyxy(boxes)

    for box, label in zip(boxes, labels):
        cls = int(label.item())
        label = names[cls]
        annotator.box_label(box, label, color=colors(cls, True))
        im0 = annotator.result()
    cv2.imshow('input', im0)
    cv2.waitKey(0)  # 1 millisecond
