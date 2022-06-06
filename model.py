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

class Conv2dMod(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-6, groups=1, demodulation=True):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels // groups, kernel_size, kernel_size, dtype=torch.float32))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # initialize weight
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.demodulation = demodulation
        self.groups = groups

    def forward(self, x, y):
        # x: (batch_size, input_channels, H, W)
        # y: (batch_size, output_channels)
        # self.weight: (output_channels, input_channels, kernel_size, kernel_size)
        N, C, H, W = x.shape

        # reshape weight
        w1 = y[:, None, :, None, None]
        w1 = w1.swapaxes(1, 2)
        w2 = self.weight[None, :, :, :, :]
        # modulate
        weight = w2 * (w1 + 1)

        # demodulate
        if self.demodulation:
            d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d
        # weight: (batch_size, output_channels, input_channels, kernel_size, kernel_size)

        # reshape
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N, *ws)


        # padding
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='replicate')

        # convolution
        x = F.conv2d(x, weight, stride=1, padding=0, groups=N * self.groups)
        x = x.reshape(N, self.output_channels, H, W)

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

# test
e = StyleEncoder()
img = torch.randn(1, 3, 256, 256)
out = e(img)
print(out.shape)
