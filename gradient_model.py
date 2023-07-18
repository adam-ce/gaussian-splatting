import torch
import torch.nn as nn
import torch.nn.functional as F

def make_unet_double_convolution(n_in_channels: int, n_hidden_channels: int, n_out_channels: int) -> nn.Sequential:
    return nn.Sequential(
            nn.Conv2d(n_in_channels, n_hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hidden_channels, n_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_out_channels),
            nn.ReLU(inplace=True)
        )

def make_unet_contraction(n_in_channels: int, n_out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.MaxPool2d(2), make_unet_double_convolution(n_in_channels, n_out_channels, n_out_channels))

def make_unet_expansion(n_in_channels: int, n_out_channels: int) -> nn.Sequential:
    class Expansion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.convolution = make_unet_double_convolution(n_in_channels, n_out_channels, n_out_channels)
        def forward(self, x: torch.Tensor, skip_connection: torch.Tensor):
            x = self.upsample(x)
            x = torch.cat((x, skip_connection), dim=1)
            x = self.convolution(x)
            return x
    return Expansion()

def make_unet(n_in_channels: int, n_out_channels: int) -> nn.Sequential:
    class Unet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            n_channels_base = 64
            self.in_convolution = make_unet_double_convolution(n_in_channels, n_channels_base, n_channels_base)
            self.contraction_1 = make_unet_contraction(n_channels_base, n_channels_base * 2)
            self.contraction_2 = make_unet_contraction(n_channels_base * 2, n_channels_base * 4)
            self.contraction_3 = make_unet_contraction(n_channels_base * 4, n_channels_base * 8)
            self.contraction_4 = make_unet_contraction(n_channels_base * 8, n_channels_base * 16)
            self.contraction_5 = make_unet_contraction(n_channels_base * 16, n_channels_base * 32)
            self.expansion_1 = make_unet_expansion(n_channels_base * 48, n_channels_base * 16)
            self.expansion_2 = make_unet_expansion(n_channels_base * 24, n_channels_base * 8)
            self.expansion_3 = make_unet_expansion(n_channels_base * 12, n_channels_base * 4)
            self.expansion_4 = make_unet_expansion(n_channels_base * 6, n_channels_base * 2)
            self.expansion_5 = make_unet_expansion(n_channels_base * 3, n_channels_base)
            self.out_convolution = make_unet_double_convolution(n_channels_base, n_channels_base, n_out_channels)
            
        def forward(self, x: torch.Tensor):
            x1 = self.in_convolution(x)
            x2 = self.contraction_1(x1)
            x3 = self.contraction_2(x2)
            x4 = self.contraction_3(x3)
            x5 = self.contraction_4(x4)
            x = self.contraction_5(x5)
            x = self.expansion_1(x, x5)
            x = self.expansion_2(x, x4)
            x = self.expansion_3(x, x3)
            x = self.expansion_4(x, x2)
            x = self.expansion_5(x, x1)
            x = self.out_convolution(x)
            return x
    
    return Unet()
