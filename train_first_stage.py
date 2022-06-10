import torch
import torch.optim as optim
import torchvision.transforms as transforms

from dataset import ImageDataset
from model import *
from to_lineart import to_lineart
from tqdm import tqdm
import os

NUM_EPOCH = 100
BATCH_SIZE = 8
IMAGE_SIZE = 256

from config_first_stage import *
unet = UNet(**unet_configs)
style_encoder = ConvNeXt(**style_encoder_configs)
discriminator = ConvNeXt(**discriminator_configs)

if os.path.exists("fs_unet.pt"):
    unet.load_state_dict(torch.load("fs_unet.pt"))
if os.path.exists("fs_style.pt"):
    style_encoder.load_state_dict(torch.load("fs_style.pt"))
if os.path.exists("fs_discriminator.pt"):
    discriminator.load_state_dict(torch.load("fs_discriminator.pt"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unet.to(device)
style_encoder.to(device)
discriminator.to(device)

ds = ImageDataset(["/mnt/d/local-develop/lineart2image_data_generator/colorized_1024x/"], max_len=100)
ds.set_size(IMAGE_SIZE)

aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation((-180, 180))]),
        transforms.RandomApply([transforms.RandomCrop((round(IMAGE_SIZE * 0.75), round(IMAGE_SIZE * 0.75)))]),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])

bar_epoch = tqdm(total=len(ds) * NUM_EPOCH, position=1)
bar_batch = tqdm(total=len(ds), position=0)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
optim_senc = optim.RAdam(style_encoder.parameters(), lr=1e4)
optim_unet = optim.RAdam(unet.parameters(), lr=1e4)
optim_disc = optim.RAdam(discriminator.parameters(), lr=1e4)

for i in range(NUM_EPOCH):
    for j, img in enumerate(dl):
        N = img.shape[0]
        

        bar_epoch.update(N)
        bar_batch.update(N)
    bar_batch.reset()
