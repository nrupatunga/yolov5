import torch

from models.yolo import Model

model = Model('./models/yolov5n.yaml', ch=3, nc=6,
              anchors=3)
torch.onnx.export(model=model,
                  args=torch.zeros(1, 3, 288, 224),
                  f='yolov5n-netron.onnx', opset_version=11)
