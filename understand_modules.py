import torch

from models.common import C3, SPPF

torch.onnx.export(model=SPPF(128, 128, 5), args=torch.zeros(1, 128, 80, 80),
                  f='SPPF.onnx')
torch.onnx.export(model=C3(128, 128, 1), args=torch.zeros(1, 128, 80, 80),
                  f='C3.onnx')
