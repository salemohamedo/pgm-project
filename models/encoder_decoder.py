import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncodingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encode = nn.Linear(2, dim)

    def forward(self, x):
        b, c, h, w = x.shape

        # Create 2d positional grid
        grid_x, grid_y  = torch.meshgrid(torch.linspace(-1, 1, w), torch.linspace(-1, 1, h))
        pos_encoding = torch.stack([grid_x, grid_y], dim=-1).to(x.device)

        pos_embed = self.encode(pos_encoding)

        # (h, w, dim) --> (dim, h, w) --> (b, dim, h, w)
        pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)

        x = x + pos_embed

        return x

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, c_hid, latent_dim):
        super().__init__()
        self.scale_factor = 4.0
        self.latent_dim = latent_dim
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, c_hid, kernel_size=3, padding=1, stride=2),
            PositionalEncodingLayer(c_hid),
            nn.SiLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), 
            nn.SiLU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
        )
        self.project = nn.Sequential(
            nn.Linear(2*c_hid*4*4, 4*c_hid),
            nn.SiLU(),
            nn.Linear(4*c_hid, 2*latent_dim)
            )

    def forward(self, x):
        x = self.encode(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.project(x)

        mean, log_std =  x.split(self.latent_dim, dim=-1)

        # They use this normalization (Stability Issue)
        log_std = torch.tanh(log_std / self.scale_factor) * self.scale_factor

        return mean, log_std


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, c_hid, latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), 
            PositionalEncodingLayer(2*c_hid),
            nn.SiLU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            PositionalEncodingLayer(2*c_hid),
            nn.SiLU(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), 
            PositionalEncodingLayer(c_hid),
            nn.SiLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            PositionalEncodingLayer(c_hid),
            nn.SiLU(),
            nn.ConvTranspose2d(c_hid, in_channels, kernel_size=3, output_padding=1, padding=1, stride=2), 
            nn.Tanh() 
        )
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 4*c_hid),
            nn.SiLU(),
            nn.Linear(4*c_hid, 2*c_hid*4*4),
            )

    def forward(self, x):
        x = self.project(x)
        x = x.view(x.shape[0], -1, 4, 4)
        x = self.decode(x)

        return x



class EncoderBock(nn.Module):
    def __init__(self, in_channels, out_channels, use_positional_encoding=False):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)]
        if use_positional_encoding:
            layers += [PositionalEncodingLayer(out_channels)]

        layers += [nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU()]

        self.embed = nn.Sequential(*layers)

    def forward(self, x):
        return self.embed(x)

  
class ComplexEncoder(nn.Module):
    def __init__(self, in_channels, c_hid, latent_dim, stocastic=True):
        super().__init__()

        # Learnable scaling factor for each dimension. Constant factor of 4.0 for all dim is used in Simple Encoder
        self.scale_factor = nn.Parameter(torch.zeros(latent_dim,))
        self.latent_dim = latent_dim
        self.stocastic = stocastic

        self.encode = nn.Sequential(
            EncoderBock(in_channels, out_channels=c_hid, use_positional_encoding=True),
            EncoderBock(c_hid, out_channels=c_hid),
            EncoderBock(c_hid, out_channels=c_hid),
            EncoderBock(c_hid, out_channels=c_hid)
        )

        latent_dim = 2*latent_dim if self.stocastic else latent_dim
        self.project = nn.Sequential(
            nn.Linear(4*4*c_hid, 4*c_hid),
            nn.LayerNorm(4*c_hid),
            nn.SiLU(),
            nn.Linear(4*c_hid, latent_dim)
            )

    def forward(self, x):
        x = self.encode(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.project(x)

        if not self.stocastic:
            return x

        mean, log_std = x.split(self.latent_dim, dim=-1)

        # For prediction stability
        scale_factor = F.softplus(self.scale_factor)
        log_std = torch.tanh(log_std / scale_factor) * scale_factor

        return mean, log_std


class ResidualBock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.embed = nn.Sequential(
                        nn.BatchNorm2d(channels),
                        nn.SiLU(),
                        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                        PositionalEncodingLayer(channels),
                        nn.BatchNorm2d(channels),
                        nn.SiLU(),
                        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
                    )

    def forward(self, x):
        return self.embed(x) + x


class ComplexDecoder(nn.Module):
    def __init__(self, in_channels, c_hid, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 4*c_hid),
            nn.LayerNorm(4*c_hid),
            nn.SiLU(),
            nn.Linear(4*c_hid, c_hid*4*4),
            )

        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True),
            ResidualBock(c_hid),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True),
            ResidualBock(c_hid),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True),
            ResidualBock(c_hid),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True),
            ResidualBock(c_hid),

            #preactivation
            nn.BatchNorm2d(c_hid),
            nn.SiLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_hid),
            nn.SiLU(),
            nn.Conv2d(c_hid, in_channels, kernel_size=1, stride=1),
            nn.Tanh()
        )

        
    def forward(self, x):
        x = self.project(x)
        x = x.view(x.shape[0], -1, 4, 4)
        x = self.decode(x)

        return x

if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    pos_e = PositionalEncodingLayer(3)
    y = pos_e(x)
    assert x.shape == y.shape
    print(y.shape)

    encoder = SimpleEncoder(in_channels=3, c_hid=16, latent_dim=16)
    decoder = SimpleDecoder(in_channels=3, c_hid=16, latent_dim=16)
    y = decoder(encoder(x)[0])
    assert x.shape == y.shape
    print(y.shape)

    x = torch.randn(2, 3, 64, 64)
    encoder = ComplexEncoder(in_channels=3, c_hid=16, latent_dim=16, stocastic=True)
    decoder = ComplexDecoder(in_channels=3, c_hid=16, latent_dim=16)
    y = decoder(encoder(x)[0])
    assert x.shape == y.shape
    print(y.shape)
    


