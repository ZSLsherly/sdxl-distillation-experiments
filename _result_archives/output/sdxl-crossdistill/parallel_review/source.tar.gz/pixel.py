"""Batch-capable copy of the existing frozen DINOv2 + RGB pixel discriminator."""
import math
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
from transformers import Dinov2Model


class RGBDiscriminator(nn.Module):
    def __init__(self, text_dim=1280):
        super().__init__()
        layers, channels = [], 3
        for width, stride in ((48, 2), (96, 2), (192, 2), (192, 1)):
            layers.extend([spectral_norm(nn.Conv2d(channels, width, 4, stride, 1)), nn.LeakyReLU(.2)])
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = spectral_norm(nn.Conv2d(channels, 1, 4, 1, 1))
        self.text = spectral_norm(nn.Linear(text_dim, channels))

    def forward(self, rgb, text):
        features = self.features(rgb.float())
        text = torch.nn.functional.normalize(text.float(), dim=-1)
        projection = (features.mean((2, 3)) * self.text(text)).sum(1) / math.sqrt(features.shape[1])
        return self.head(features).mean((1, 2, 3)) + projection


class PixelDiscriminator(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(path, local_files_only=True, attn_implementation="eager").eval().requires_grad_(False)
        channels = self.backbone.config.hidden_size
        self.heads = nn.ModuleList([
            nn.Sequential(spectral_norm(nn.Conv2d(channels, 128, 1)), nn.LeakyReLU(.2),
                          spectral_norm(nn.Conv2d(128, 1, 3, padding=1))) for _ in range(4)])
        self.rgb = RGBDiscriminator()
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def set_heads(self, train):
        self.heads.requires_grad_(train)
        self.rgb.requires_grad_(train)
        self.train(train)

    def trainable(self):
        return list(self.heads.parameters()) + list(self.rgb.parameters())

    def forward(self, rgb, text, return_parts=False):
        x = (rgb[:, :, 16:240, 16:240] / 2 + .5 - self.mean) / self.std

        def features(z):
            hidden = self.backbone(z, output_hidden_states=True).hidden_states
            return tuple(hidden[i][:, 1:].transpose(1, 2).reshape(z.shape[0], -1, 16, 16) for i in (3, 6, 9, 12))

        hidden = checkpoint(features, x, use_reentrant=False) if x.requires_grad else features(x)
        heads = torch.stack([head(z).mean((1, 2, 3)) for head, z in zip(self.heads, hidden)])
        dino, rgb_score = heads.mean(0), self.rgb(rgb, text)
        total = .5 * (dino + rgb_score)
        if return_parts:
            return {"total": total, "dino": dino, "rgb": rgb_score, "dino_heads": heads}
        return total
