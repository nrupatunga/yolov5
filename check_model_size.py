import cv2
import numpy as np
import torch

from models.yolo import Model

# verify the forward function to understand the architecture
img = cv2.imread('./data/images/zidane.jpg')
img = cv2.resize(img, (288, 288))
img = np.transpose(img, (2, 0, 1)).astype(np.float32)
img = img / 255.
img_t = torch.from_numpy(img)[None]

model = Model('./models/yolov5-shufflenet.yaml', ch=3, nc=6, anchors=3)
torch.save(model.state_dict(), 'shufflenet.pth')
output = model(img_t, profile=True)
print('-------------------------------------------')
model = Model('./models/yolov5nC3NB.yaml', ch=3, nc=6, anchors=3)
torch.save(model.state_dict(), 'c3nb.pth')
output = model(img_t, profile=True)
print('-------------------------------------------')
model = Model('./models/yolov5n.yaml', ch=3, nc=6, anchors=3)
torch.save(model.state_dict(), 'c3nb.pth')
output = model(img_t, profile=True)
