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
        self.gelu = nn.GELU()
        self.c3 = nn.Conv2d(dim_ffn, channels, 1, 1, 0)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = self.gelu(x)
        x = self.c3(x)
        return x + res

# Input: [N, input_channels, H, W]
# Output: [N, output_features]
class ConvNeXt(nn.Module):
    def __init__(self, input_channels=3, stages=[3, 3, 3, 3], channels=[32, 64, 128, 256], output_features=256, minibatch_std=False):
        super().__init__()
        self.stem = nn.Conv2d(input_channels, channels[0], 4, 4, 0)
        seq = []
        if minibatch_std:
            self.out_linear = nn.Sequential(nn.Linear(channels[-1]+1, output_features), nn.Linear(output_features, output_features))
        else:
            self.out_linear = nn.Linear(channels[-1], output_features)
        self.mb_std = minibatch_std
        for i, (l, c) in enumerate(zip(stages, channels)):
            for _ in range(l):
                seq.append(ConvNeXtBlock(c))
            if i != len(stages)-1:
                seq.append(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0))
                seq.append(ChannelNorm(channels[i+1]))
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        x = self.stem(x)
        x = self.seq(x)
        x = torch.mean(x,dim=[2,3], keepdim=False)
        if self.mb_std:
            mb_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1)
            x = torch.cat([x, mb_std], dim=1)
            x = self.out_linear(x)
        else:
            x = self.out_linear(x)
        return x

class UNetBlock(nn.Module):
    def __init__(self, stage, ch_conv):
        super().__init__()
        self.stage = stage
        self.ch_conv = ch_conv

# UNet with style
class StyleUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, stages=[3, 3, 9, 3], channels=[32, 64, 128, 256], style_dim=512, tanh=True):
        super().__init__()
        self.encoder_first = nn.Conv2d(input_channels, channels[0], 4, 4, 0)

        self.decoder_last = nn.Sequential(
                nn.ConvTranspose2d(channels[0], output_channels*6, 4, 4, 0),
                nn.GELU(),
                nn.Conv2d(output_channels*6, output_channels*3, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(output_channels*3, output_channels, 3, 1, 1, padding_mode='replicate'),
                )
        self.style_affine = nn.Linear(style_dim, channels[-1]) 
        self.tanh = nn.Tanh() if tanh else nn.Identity()
        self.encoder_stages = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])
        for i, (l, c) in enumerate(zip(stages, channels)):
            enc_stage = nn.Sequential(*[ConvNeXtBlock(c) for _ in range(l)])
            enc_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0), ChannelNorm(channels[i+1]))
            dec_stage = nn.Sequential(*[ConvNeXtBlock(c) for _ in range(l)])
            dec_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.ConvTranspose2d(channels[i+1], channels[i], 2, 2, 0), ChannelNorm(channels[i]))
            self.encoder_stages.append(UNetBlock(enc_stage, enc_ch_conv))
            self.decoder_stages.insert(0, UNetBlock(dec_stage, dec_ch_conv))

    def forward(self, x, style):
        x = self.encoder_first(x)
        skips = []
        for l in self.encoder_stages:
            x = l.stage(x)
            skips.insert(0, x)
            x = l.ch_conv(x)
        x += self.style_affine(style).unsqueeze(2).unsqueeze(2).expand(-1, -1, x.shape[2], x.shape[3])
        for i, (l, s) in enumerate(zip(self.decoder_stages, skips)):
            x = l.ch_conv(x)
            x = l.stage(x + s)
        x = self.decoder_last(x)
        x = self.tanh(x)
        return x

class DraftGAN(nn.Module):
    def __init__(self, style_dim=256):
        super().__init__()
        self.style_encoder = ConvNeXt(output_features=style_dim, minibatch_std=False, stages=[2, 2, 2, 2], channels=[16, 32, 64, 128])
        self.discriminator = ConvNeXt(output_features=1, minibatch_std=True, stages=[3,3,3,3], channels=[24, 32, 64, 128])
        self.colorizer = StyleUNet(style_dim=style_dim)
