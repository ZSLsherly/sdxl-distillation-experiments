"""Matched discriminator/generator objectives. R1 remains in the training loop."""
import torch
from torch.nn import functional as F


def discriminator_loss(real, fake, kind='hinge'):
    if kind=='hinge':
        return torch.relu(1-real).mean()+torch.relu(1+fake).mean()
    if kind=='logistic':
        return F.softplus(-real).mean()+F.softplus(fake).mean()
    raise ValueError(f'Unknown GAN loss: {kind}')


def generator_loss(fake, kind='hinge'):
    if kind=='hinge':
        return -fake.mean()
    if kind=='logistic':
        return F.softplus(-fake).mean()
    raise ValueError(f'Unknown GAN loss: {kind}')
