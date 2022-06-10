import torch
import torch.optim as optim
import torchvision.transforms as transforms

from dataset import ImageDataset
from model import *
from to_lineart import to_lineart
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

NUM_EPOCH = 1000
MAX_DATASET_LEN = 200
IMAGE_SIZE = 256
test_dir = "./fs_test/"

from config_first_stage import *
unet = UNet(**unet_configs)
style_encoder = ConvNeXt(**style_encoder_configs)
discriminator = ConvNeXt(**discriminator_configs)

if os.path.exists("fs_unet.pt"):
    unet.load_state_dict(torch.load("fs_unet.pt"))
    print("Loaded UNet")
if os.path.exists("fs_style.pt"):
    style_encoder.load_state_dict(torch.load("fs_style.pt"))
    print("Loaded style encoder")


if not os.path.exists(test_dir):
    os.mkdir(test_dir)

ds = ImageDataset(["/mnt/d/local-develop/lineart2image_data_generator/colorized_1024x/"], max_len=MAX_DATASET_LEN, chache_dir="./test_chache/")
ds.set_size(IMAGE_SIZE)
dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
style_dim = unet_configs["style_dim"]

with torch.no_grad():
    for i, image in enumerate(dl):
        style = style_encoder(image[0:1])
        lineart = to_lineart(image[1:2])[:, 0:1]
        out = unet(lineart, style=style)
    
        # save lineart
        path = os.path.join(test_dir, f"{i}_lineart.jpg")
        img = Image.fromarray((lineart[0].detach().expand(3, -1, -1).cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
        img = img.resize((256, 256))
        img.save(path)

        # save output
        path = os.path.join(test_dir, f"{i}_colorized.jpg")
        img = Image.fromarray((out[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
        img = img.resize((256, 256))
        img.save(path)

        # save style image
        path = os.path.join(test_dir, f"{i}_style.jpg")
        img = Image.fromarray((image[1].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
        img = img.resize((256, 256))
        img.save(path)


