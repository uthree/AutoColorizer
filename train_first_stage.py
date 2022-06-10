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
BATCH_SIZE = 16
IMAGE_SIZE = 256
MAX_DATASET_LEN = 2000
result_dir = "./fs_results/"

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
if os.path.exists("fs_discriminator.pt"):
    print("Loaded discriminator")
    discriminator.load_state_dict(torch.load("fs_discriminator.pt"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unet.to(device)
style_encoder.to(device)
discriminator.to(device)

ds = ImageDataset(["/mnt/d/local-develop/lineart2image_data_generator/colorized_1024x/"], max_len=MAX_DATASET_LEN)
ds.set_size(IMAGE_SIZE)

aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation((-180, 180))]),
        transforms.RandomApply([transforms.RandomCrop((round(IMAGE_SIZE * 0.75), round(IMAGE_SIZE * 0.75)))]),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])

MSE = torch.nn.MSELoss()
bar_epoch = tqdm(total=len(ds) * NUM_EPOCH, position=1)
bar_batch = tqdm(total=len(ds), position=0)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
style_encoder.to(device)
unet.to(device)
discriminator.to(device)

optim_senc = optim.RAdam(style_encoder.parameters(), lr=1e-4)
optim_unet = optim.RAdam(unet.parameters(), lr=1e-4)
optim_disc = optim.RAdam(discriminator.parameters(), lr=1e-4)

# create result directory
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

for i in range(NUM_EPOCH):
    for j, img in enumerate(dl):
        N = img.shape[0]
        img = img.to(device)
        img = aug(img)
        lineart = to_lineart(img)[:, 0:1] # convert to lineart
        style_input = aug(img)
        # train generator
        optim_senc.zero_grad()
        optim_unet.zero_grad()
        style = style_encoder(style_input)
        fake = unet(lineart, style=style)
        g_adv_loss = MSE(discriminator(fake) ,torch.zeros(N, 1, device=device)) 
        g_mse_loss = MSE(fake, img)
        g_loss = g_adv_loss + g_mse_loss
        g_loss.backward()
        optim_senc.step()
        optim_unet.step()

        # train discriminator
        optim_disc.zero_grad()
        fake = fake.detach()
        logit_fake = discriminator(fake)
        logit_real = discriminator(img)
        d_loss_f = MSE(logit_fake ,torch.ones(N, 1, device=device))
        d_loss_r = MSE(logit_real ,torch.zeros(N, 1, device=device))
        d_loss = d_loss_f + d_loss_r
        d_loss.backward()
        optim_disc.step()

        # set bar description
        bar_batch.set_description(desc=f"G.Loss:{g_loss.item():.4f}(MSE: {g_mse_loss.item():.4f}, Adv.:{g_adv_loss.item():.4f}) D.Loss: {d_loss.item():.4f}")

        if j % 1000 == 0:
            # save lineart
            path = os.path.join(result_dir, f"{i}_{j}_lineart.jpg")
            img = Image.fromarray((lineart[0].detach().expand(3, -1, -1).cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
            img = img.resize((256, 256))
            img.save(path)

            # save painted image
            path = os.path.join(result_dir, f"{i}_{j}_colorized.jpg")
            img = Image.fromarray((fake[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
            img = img.resize((256, 256))
            img.save(path)

            # save model
            torch.save(unet.state_dict(), "./fs_unet.pt")
            torch.save(style_encoder.state_dict(), "./fs_style.pt")
            torch.save(discriminator.state_dict(), "./fs_discriminator.pt")

        bar_epoch.update(N)
        bar_batch.update(N)
    bar_batch.reset()
