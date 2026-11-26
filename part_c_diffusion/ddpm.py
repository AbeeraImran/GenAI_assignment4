import torch
from diffusers import UNet2DModel

T = 1000

def make_unet():
    return UNet2DModel(
        sample_size=28, in_channels=1, out_channels=1, layers_per_block=2,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=40)

def make_schedule(device):
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar

def add_noise(x0, t, eps, alpha_bar):
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

@torch.no_grad()
def sample(unet, labels, betas, alphas, alpha_bar, keep=()):
    n = len(labels)
    x = torch.randn(n, 1, 28, 28, device=labels.device)
    saved = {}
    for t in range(T - 1, -1, -1):
        if t in keep:
            saved[t] = x.clone()
        t_batch = torch.full((n,), t, device=labels.device, dtype=torch.long)
        eps = unet(x, t_batch, class_labels=labels).sample
        x = (x - (1 - alphas[t]) / torch.sqrt(1 - alpha_bar[t]) * eps) / torch.sqrt(alphas[t])
        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)
    x = x.clamp(-1, 1)
    if 0 in keep:
        saved[0] = x.clone()
    return x, saved
