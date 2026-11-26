import torch
import torch.nn as nn

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(192, 192)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 192) * 0.02)
        self.encoder = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=192, nhead=3, dim_feedforward=768,
                                       dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(6)])
        self.enc_norm = nn.LayerNorm(192)
        self.dec_embed = nn.Linear(192, 128)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 128) * 0.02)
        self.dec_pos_embed = nn.Parameter(torch.randn(1, 64, 128) * 0.02)
        self.decoder = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512,
                                       dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(2)])
        self.dec_pred = nn.Linear(128, 192)

    def random_mask(self, n, device):
        noise = torch.rand(n, 64, device=device)
        ids_keep = noise.argsort(dim=1)[:, :16]
        mask = torch.ones(n, 64, device=device)
        mask.scatter_(1, ids_keep, 0.0)
        return ids_keep, mask

    def encode_visible(self, patches, ids_keep):
        x = self.patch_embed(patches) + self.pos_embed
        idx = ids_keep.unsqueeze(-1).expand(-1, -1, 192)
        x = torch.gather(x, 1, idx)
        x = self.encoder(x)
        return self.enc_norm(x)

    def decode(self, x, ids_keep):
        n = x.shape[0]
        x = self.dec_embed(x)
        full = self.mask_token.repeat(n, 64, 1)
        idx = ids_keep.unsqueeze(-1).expand(-1, -1, 128)
        full = full.scatter(1, idx, x)
        full = full + self.dec_pos_embed
        full = self.decoder(full)
        return self.dec_pred(full)

    def forward(self, patches):
        ids_keep, mask = self.random_mask(patches.shape[0], patches.device)
        x = self.encode_visible(patches, ids_keep)
        pred = self.decode(x, ids_keep)
        return pred, mask

    def features(self, patches):
        x = self.patch_embed(patches) + self.pos_embed
        x = self.encoder(x)
        x = self.enc_norm(x)
        return x.mean(dim=1)

def mae_loss(pred, patches, mask):
    err = ((pred - patches) ** 2).mean(dim=-1)
    return (err * mask).sum() / mask.sum()
