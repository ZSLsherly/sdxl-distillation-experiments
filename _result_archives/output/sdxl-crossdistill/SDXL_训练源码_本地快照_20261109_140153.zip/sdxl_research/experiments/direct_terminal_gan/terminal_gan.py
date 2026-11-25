"""Fourth-step GAN routing shared by discriminator and generator."""
def terminal_input(x0, refine=None):
    # The view shares values with DM, but its gradient contains only the GAN branch.
    return x0.view_as(x0) if refine is None else refine(x0.detach())


def backward_terminal(dm, x0, refine, score_loss, weight, accumulation, probe=None):
    if refine is not None:
        (dm / accumulation).backward()
    gan_input = terminal_input(x0, refine)
    if probe is not None:
        gan_input.register_hook(probe)
    gan = score_loss(gan_input)
    loss = weight * gan
    if refine is None:
        loss = dm + loss
    (loss / accumulation).backward()
    return gan.detach()
