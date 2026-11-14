"""Three-reduction U-Net intensity candidate with full-resolution skips.

Adds a 128-channel bottleneck and the matching 64-channel decoder block to
the frozen shallow model. The comparison changes capacity and compute as
well as context depth; it is not an isolated downsampling ablation.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .shallow_unet import _block


class DeeperUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = _block(1, 16)
        self.enc2 = _block(16, 32)
        self.enc3 = _block(32, 64)
        self.bottleneck = _block(64, 128)
        self.dec3 = _block(192, 64)
        self.dec2 = _block(96, 32)
        self.dec1 = _block(48, 16)
        self.head = nn.Conv2d(16, 1, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 8:
            raise ValueError('Expected B,1,H,W with spatial sizes at least eight')
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        z = self.bottleneck(F.max_pool2d(e3, 2))
        z = self.dec3(torch.cat((F.interpolate(z, size=e3.shape[-2:],
                              mode='bilinear', align_corners=False), e3), dim=1))
        z = self.dec2(torch.cat((F.interpolate(z, size=e2.shape[-2:],
                              mode='bilinear', align_corners=False), e2), dim=1))
        z = self.dec1(torch.cat((F.interpolate(z, size=e1.shape[-2:],
                              mode='bilinear', align_corners=False), e1), dim=1))
        return self.head(z)
