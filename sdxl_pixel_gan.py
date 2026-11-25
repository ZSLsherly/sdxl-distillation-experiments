"""Small conditional RGB PatchGAN and differentiable native-scale VAE tiles."""
import math
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint

class PixelPatchDiscriminator(nn.Module):
    def __init__(self, text_dim=1280):
        super().__init__()
        layers=[];c=3
        for width,stride in [(48,2),(96,2),(192,2),(192,1)]:
            layers += [spectral_norm(nn.Conv2d(c,width,4,stride,1)),nn.LeakyReLU(.2)]
            c=width
        self.features=nn.Sequential(*layers)
        self.head=spectral_norm(nn.Conv2d(c,1,4,1,1))
        self.text=spectral_norm(nn.Linear(text_dim,c))
    def forward(self, rgb, text):
        f=self.features(rgb.float())
        # Unit text normalization keeps embedding magnitude out of the GAN game.
        text=torch.nn.functional.normalize(text.float(),dim=-1)
        projection=(f.mean((2,3))*self.text(text)).sum(1)/math.sqrt(f.shape[1])
        return self.head(f).mean((1,2,3))+projection

def decode_tile(vae, latent, xy=(40,40), grad=False):
    """48 latent pixels -> 384 RGB pixels; discard a 64 RGB pixel halo.

    Decoder GroupNorm is spatial, so tiled decoding is an approximation to full
    decoding, even away from boundaries. Both real and fake use the same rule.
    No tanh/clamp is applied in training: saturated images must retain gradients.
    """
    y,x=xy;tile=latent[:,:,y:y+48,x:x+48]/vae.config.scaling_factor
    def fn(z):return vae.decode(z.float(),return_dict=False)[0][:,:,64:320,64:320]
    return checkpoint(fn,tile,use_reentrant=False) if grad else fn(tile)

def discriminator_loss(real_logits,fake_logits):
    return torch.nn.functional.relu(1-real_logits).mean()+torch.nn.functional.relu(1+fake_logits).mean()

def average_gradients(model, world):
    if world>1:
        import torch.distributed as dist
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad);p.grad.div_(world)
