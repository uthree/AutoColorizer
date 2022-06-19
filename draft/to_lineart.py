import torch
import torch.nn as nn
import torch.nn.functional as F

def to_lineart(image, filter_size=3, add_grayscale=False):
    w = torch.ones(image.shape[1] ,1, filter_size, filter_size) / (filter_size**2)
    w = w.to(image.device)
    neighbor = F.conv2d(F.pad(image, pad=(filter_size//2,)*4, mode='replicate'), w, groups=image.shape[1])
    abs_diff = torch.abs(neighbor - image)
    out = abs_diff.mean(dim=1, keepdim=True)
    out = out.expand(-1, image.shape[1], -1, -1)
    out = F.conv2d(out, w, padding='same', groups=image.shape[1])
    out = (out / torch.max(torch.abs(out))) * 2
    out = torch.clamp(out-0.1, min=0)
    out = out - 1
    out = -out
    if add_grayscale:
        out = (out + image.mean(dim=1, keepdim=True)) / 2
    out = torch.clamp(out, min=-1, max=1)
    return out

