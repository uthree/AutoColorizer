import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

def to_lineart(image, filter_size=3, add_grayscale=False):
    w = torch.ones(image.shape[1] ,1, filter_size, filter_size) / (filter_size**2)
    neighbor = F.conv2d(image, w, padding='same', groups=image.shape[1])
    abs_diff = torch.abs(neighbor - image)
    out = abs_diff.mean(dim=1, keepdim=True)
    out = out.expand(-1, image.shape[1], -1, -1)
    out = F.conv2d(out, w, padding='same', groups=image.shape[1])
    print(torch.max(torch.abs(out)))
    out = out / torch.max(torch.abs(out))
    out = out - 1
    out = - out
    if add_grayscale:
        out = (out + image.mean(dim=1, keepdim=True)) / 2
    return out

# test

img = Image.open('./test.jpg')
img = (np.array(img).transpose(2, 0, 1).astype(float) - 127.5) / 127.5
img = torch.FloatTensor(img)
img = img.unsqueeze(0)
img = to_lineart(img)

# save image
img = img[0].cpu().numpy() * 127.5 + 127.5
img = img.astype(np.int8)
img = img.transpose(1,2,0)
img = Image.fromarray(img, mode="RGB")
img.save("out.jpg")

