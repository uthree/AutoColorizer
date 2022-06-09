import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(ChannelNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x): # x: [N, C, H, W]
        m = x.mean(dim=1, keepdim=True)
        s = ((x - m) ** 2).mean(dim=1, keepdim=True)
        x = (x - m) * torch.rsqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, dim_ffn=None, kernel_size=7):
        super(ConvNeXtBlock, self).__init__()
        if dim_ffn == None:
            dim_ffn = channels * 4
        self.c1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, padding_mode='replicate', groups=channels)
        self.norm = ChannelNorm(channels)
        self.c2 = nn.Conv2d(channels, dim_ffn, 1, 1, 0)
        self.gelu = nn.LeakyReLU(0.2)
        self.c3 = nn.Conv2d(dim_ffn, channels, 1, 1, 0)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = self.gelu(x)
        x = self.c3(x)
        return x + res

# Input: [N, input_channels, 256, 256]
# Output: [N, style_dim]
class StyleEncoder(nn.Module):
    def __init__(self, input_channels=3, stages=[2, 2, 2, 2], channels=[16, 32, 64, 128], style_dim=512):
        super().__init__()
        self.stem = nn.Conv2d(input_channels, channels[0], 4, 4, 0)
        seq = []
        self.to_style = nn.Linear(channels[-1], style_dim)
        for i, (l, c) in enumerate(zip(stages, channels)):
            for _ in range(l):
                seq.append(ConvNeXtBlock(c))
            if i != len(stages)-1:
                seq.append(nn.AvgPool2d(kernel_size=2))
                seq.append(nn.Conv2d(channels[i], channels[i+1], 1, 1, 0))
                seq.append(ChannelNorm(channels[i+1]))
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        x = self.stem(x)
        x = self.seq(x)
        x = torch.mean(x,dim=[2,3], keepdim=False)
        x = self.to_style(x)
        return x

class UNetBlock(nn.Module):
    def __init__(self, stage, ch_conv):
        super().__init__()
        self.stage = stage
        self.ch_conv = ch_conv

# UNet
class UNet(nn.Module):
    def __init__(self, input_channels=4, output_channels=3, stages=[2,2,2,2], channels=[32, 64, 128, 256], stem=True, style=False, style_dim=512):
        super().__init__()
        if stem:
            self.encoder_first = nn.Conv2d(input_channels, channels[0], 4, 4, 0)
            self.decoder_last = nn.ConvTranspose2d(channels[0], output_channels, 4, 4, 0)
        else:
            self.encoder_first = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
            self.decoder_last = nn.Conv2d(channels[0], output_channels, 1, 1, 0)
        self.encoder_stages = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])
        self.style = style
        self.style_affine = nn.Linear(style_dim, channels[-1]) if style else nn.Identity()
        for i, (l, c) in enumerate(zip(stages, channels)):
            enc_stage = nn.Sequential(*[ConvNeXtBlock(c) for _ in range(l)])
            enc_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Conv2d(channels[i], channels[i+1], 1, 1, 0), nn.AvgPool2d(kernel_size=2))
            dec_stage = nn.Sequential(*[ConvNeXtBlock(c) for _ in range(l)])
            dec_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Conv2d(channels[i+1], channels[i], 1, 1, 0), nn.Upsample(scale_factor=2))
            self.encoder_stages.append(UNetBlock(enc_stage, enc_ch_conv))
            self.decoder_stages.insert(0, UNetBlock(dec_stage, dec_ch_conv))

    def forward(self, x, style=None):
        x = self.encoder_first(x)
        skips = []
        for l in self.encoder_stages:
            x = l.stage(x)
            skips.insert(0, x)
            x = l.ch_conv(x)
        if self.style:
            x += self.style_affine(style).unsqueeze(2).unsqueeze(2).expand(-1, -1, x.shape[2], x.shape[3])
        for l, s in zip(self.decoder_stages, skips):
            x = l.ch_conv(x)
            x = l.stage(x + s)
        x = self.decoder_last(x)
        return x


