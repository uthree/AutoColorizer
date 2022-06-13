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

MAX_DATASET_LEN = 100
IMAGE_SIZE = 256
test_dir = "./test/"

GAN = DraftGAN()
if os.path.exists("model.pt"):
    GAN.load_state_dict(torch.load("Model.pt"))
    print("Loaded Model")

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

ds = ImageDataset(["/mnt/d/local-develop/lineart2image_data_generator/colorized_256x/"], max_len=MAX_DATASET_LEN, chache_dir="./test_chache/")
ds.set_size(IMAGE_SIZE)
dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

with torch.no_grad():
    for i, image in enumerate(dl):
        style = GAN.style_encoder(image[0:1])
        lineart = to_lineart(image[1:2])[:, 0:1]
        out = GAN.colorizer(lineart, style=style)
    
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
        img = Image.fromarray((image[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
        img = img.resize((256, 256))
        img.save(path)
