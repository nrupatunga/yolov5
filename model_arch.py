"""
File: model_arch.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description:
"""
import torch
import yaml

from models.yolo import Model

nc = 6
device = 'cuda'
weights = './yolov5n.pt'

hyp = './data/hyps/hyp.scratch.yaml'

with open(hyp, errors='ignore') as f:
    hyp = yaml.safe_load(f)  # load hyps dict

ckpt = torch.load(weights, map_location=device)  # load checkpoint
__import__('pdb').set_trace()
model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get(
    'anchors')).to(device)  # create
