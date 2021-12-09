import torch

from models.yolo import Model

model = Model('./models/yolov5nC3NB.yaml', ch=3, nc=6, anchors=3)
torch.save(model.state_dict(), 'model_weights.pth')
# torch.onnx.export(model=model, args=torch.zeros(1, 3, 288, 288),
# f='model.onnx', opset_version=11)
