"""Whole-image frozen SAM features with four lightweight trainable heads."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SAMDiscriminator(nn.Module):
    def __init__(self, path=None, backbone=None):
        super().__init__()
        if backbone is None:
            from transformers import SamModel
            backbone = SamModel.from_pretrained(path, local_files_only=True, use_safetensors=True).vision_encoder
        self.backbone = backbone.eval().requires_grad_(False)
        c = backbone.config.hidden_size
        self.size = backbone.config.image_size
        if backbone.config.num_hidden_layers != 12:
            raise ValueError('This recipe selects layers 3/6/9/12 of a 12-layer SAM encoder')
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, 128, 4, 2, 1), nn.GroupNorm(32, 128), nn.SiLU(),
                          nn.Conv2d(128, 128, 4, 2, 1), nn.GroupNorm(32, 128), nn.SiLU(),
                          nn.Conv2d(128, 1, 1)) for _ in range(4)])
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).reshape(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).reshape(1,3,1,1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def set_heads(self, train):
        self.heads.requires_grad_(train)
        self.train(False)

    def trainable(self):
        return list(self.heads.parameters())

    def features(self, rgb):
        # [-1,1] VAE pixels -> SAM ImageNet normalization. No PIL/no_grad on G.
        x = F.interpolate(rgb.float(), size=(self.size,self.size), mode='bilinear', align_corners=False)
        x = ((x + 1) * .5 - self.mean) / self.std
        def encode(z):
            hidden = self.backbone(z, output_hidden_states=True, return_dict=True).hidden_states
            return tuple(hidden[i].permute(0,3,1,2).contiguous() for i in (3,6,9,12))
        return checkpoint(encode, x, use_reentrant=False) if x.requires_grad else encode(x)

    def forward(self, rgb, text=None, return_parts=False, feature_r1=False):
        features = self.features(rgb)
        if feature_r1:
            # R1 only on independent real feature leaves, not on VAE/SAM pixels.
            features = tuple(x.detach().requires_grad_(True) for x in features)
        scores = [head(x).mean((1,2,3)) for head,x in zip(self.heads,features)]
        heads = torch.stack(scores)
        total = heads.mean(0)
        r1 = total.new_zeros(())
        if feature_r1:
            derivatives = torch.autograd.grad(sum(s.sum() for s in scores), features, create_graph=True)
            r1 = torch.stack([g.square().flatten(1).sum(1).mean() for g in derivatives]).mean()
        if return_parts:
            return {'total':total, 'features':total, 'rgb':torch.zeros_like(total),
                    'heads':heads, 'r1':r1}
        return total
