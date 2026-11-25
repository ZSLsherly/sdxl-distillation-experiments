"""Frozen pretrained DINOv2 patch features plus a trainable RGB texture head.

An experimental pixel-space discriminator inspired by VFM discriminators;
not the exact SenseFlow or SAM-based Flash-DMD discriminator.
"""
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from transformers import Dinov2Model
from torch.utils.checkpoint import checkpoint
from sdxl_pixel_gan import PixelPatchDiscriminator

class PixelDinoDiscriminator(nn.Module):
    def __init__(self,path):
        super().__init__()
        self.backbone=Dinov2Model.from_pretrained(path,local_files_only=True,attn_implementation='eager').eval().requires_grad_(False)
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        c=self.backbone.config.hidden_size
        self.heads=nn.ModuleList([nn.Sequential(spectral_norm(nn.Conv2d(c,128,1)),nn.LeakyReLU(.2),spectral_norm(nn.Conv2d(128,1,3,padding=1))) for _ in range(4)])
        self.rgb=PixelPatchDiscriminator()
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self
    def set_heads(self,train):
        self.heads.requires_grad_(train);self.rgb.requires_grad_(train);self.backbone.requires_grad_(False)
        self.train(train)
    def trainable(self):
        return list(self.heads.parameters())+list(self.rgb.parameters())
    def forward(self,rgb,text,return_parts=False):
        # 256px native VAE crop -> central 224px native DINO crop; no resize/clamp.
        x=(rgb[:,:,16:240,16:240]/2+.5-self.mean)/self.std
        def features(z):
            h=self.backbone(z,output_hidden_states=True).hidden_states
            return tuple(h[i][:,1:].transpose(1,2).reshape(z.shape[0],-1,16,16) for i in (3,6,9,12))
        h=checkpoint(features,x,use_reentrant=False) if x.requires_grad else features(x)
        heads=torch.stack([head(z).mean((1,2,3)) for head,z in zip(self.heads,h)])
        projected=heads.mean(0);texture=self.rgb(rgb,text);total=.5*(projected+texture)
        if return_parts:return {'total':total,'dino':projected,'rgb':texture,'dino_heads':heads}
        return total
