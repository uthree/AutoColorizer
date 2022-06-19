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
MAX_DATASET_LEN = 20000
result_dir = "./results/"

GAN = DraftGAN()

if os.path.exists("model.pt"):
    GAN.load_state_dict(torch.load("model.pt"))
    print("Loaded Model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAN.to(device)

ds = ImageDataset(["/mnt/d/local-develop/lineart2image_data_generator/colorized_256x/"], max_len=MAX_DATASET_LEN)
ds.set_size(IMAGE_SIZE)

aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation((-180, 180))], p=0.5),
        transforms.RandomApply([transforms.RandomCrop((round(IMAGE_SIZE * 0.8), round(IMAGE_SIZE * 0.75)))], p=0.5),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])

MSE = torch.nn.MSELoss()
bar_epoch = tqdm(total=len(ds) * NUM_EPOCH, position=1)
bar_batch = tqdm(total=len(ds), position=0)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

opt_s = optim.RAdam(GAN.style_encoder.parameters(), lr=1e-4)
opt_c = optim.RAdam(GAN.colorizer.parameters(), lr=1e-4)
opt_d = optim.RAdam(GAN.discriminator.parameters(), lr=1e-4)

S = GAN.style_encoder
C = GAN.colorizer
D = GAN.discriminator

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
        opt_s.zero_grad()
        opt_c.zero_grad()
        style = S(style_input)
        fake = C(lineart, style)
        g_adv_loss = MSE(D(fake), torch.zeros(N, 1, device=device)) 
        g_mse_loss = MSE(fake, img)
        g_loss = g_mse_loss + g_adv_loss
        g_loss.backward()
        opt_s.step()
        opt_c.step()

        # train discriminator
        opt_d.zero_grad()
        fake = fake.detach()
        logit_fake = D(fake)
        logit_real = D(img)
        d_loss_f = MSE(logit_fake, torch.ones(N, 1, device=device))
        d_loss_r = MSE(logit_real, torch.zeros(N, 1, device=device))
        d_loss = d_loss_f + d_loss_r
        d_loss.backward()
        opt_d.step()

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
            torch.save(GAN.state_dict(), "./model.pt")

        bar_epoch.update(N)
        bar_batch.update(N)
    bar_batch.reset()
