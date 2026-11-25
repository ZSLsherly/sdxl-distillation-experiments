"""GAN input views; full decoding is an expensive opt-in texture control."""
import torch
from torch.utils.checkpoint import checkpoint
from batching import decode_tiles, flip_images


def decode_view(vae, latents, crops, kind='tile', grad=False, chunk_size=None):
    if kind=='tile':
        return decode_tiles(vae,latents,crops,grad=grad,chunk_size=chunk_size)
    if kind!='full':
        raise ValueError(f'Unknown GAN view: {kind}')
    # Full FP32 images, one at a time. Take the same spatial rectangle as tile
    # decoding's 48 latent pixels and 64-pixel halo, without tile GroupNorm.
    scale=2**(len(vae.config.block_out_channels)-1)
    if scale!=8 or len(crops)!=len(latents):
        raise ValueError('Full view expects SDXL VAE scale8 and one crop per image')
    views=[]
    for i,(y,x) in enumerate(crops):
        def decode(z, y=y, x=x):
            rgb=vae.decode(z.float()/vae.config.scaling_factor,return_dict=False)[0]
            return rgb[:,:,y*scale+64:y*scale+320,x*scale+64:x*scale+320]
        z=latents[i:i+1]
        views.append(checkpoint(decode,z,use_reentrant=False) if grad else decode(z))
    return torch.cat(views)


def augment_view(rgb, flips, phase, policy='d_only'):
    if phase not in ('D','G') or policy not in ('d_only','both','none'):
        raise ValueError('Unknown GAN augmentation phase or policy')
    return flip_images(rgb,flips) if policy=='both' or (policy=='d_only' and phase=='D') else rgb
