import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_dim, 3, 1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_dim, out_dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.block(x) + x
    


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_blocks, res_dim):
        super().__init__()

        k, s = 4, 2

        self.encoder = nn.Sequential(

            nn.Conv2d(in_dim, h_dim//2, k, s, padding=1),
            nn.ReLU(True),

            nn.Conv2d(h_dim//2, h_dim, k, s, padding=1),
            nn.ReLU(True),

            nn.Conv2d(h_dim, h_dim, k-1, s-1, padding=1),

            *[ResBlock(h_dim, h_dim, res_dim) for _ in range(n_res_blocks)],
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, h_dim, out_dim, n_res_blocks, res_dim):
        super().__init__()

        k, s = 4, 2

        self.decoder = nn.Sequential(

            *[ResBlock(h_dim, h_dim, res_dim) for _ in range(n_res_blocks)],
            nn.ReLU(True),

            nn.ConvTranspose2d(h_dim, h_dim//2, k, s, padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(h_dim//2, out_dim, k, s, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)


class VectorQuantizer(nn.Module):
    def __init__(self, n_emb, emb_dim, beta, device):
        super().__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.beta = beta
        self.device = device

        self.emb = nn.Embedding(n_emb, emb_dim)
        init = 1.0 / n_emb
        self.emb.weight.data.uniform_(-init, init)

    def forward(self, z):
        # (B, C, H, W) -> (BHW, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.emb_dim)

        # (z-e)^2 = z^2 + e^2 - 2ze
        distances = torch.sum(z_flat**2, dim=1, keepdim=True) + \
                    torch.sum(self.emb.weight**2, dim=1) - \
                    2 * torch.matmul(z_flat, self.emb.weight.t())

        nearest_idx = torch.argmin(distances, dim=1).unsqueeze(1)
        nearest = torch.zeros(nearest_idx.shape[0], self.n_emb).to(self.device)
        nearest.scatter_(1, nearest_idx, 1)

        zq = torch.matmul(nearest, self.emb.weight).view(z.shape)

        loss = torch.mean((zq.detach() - z)**2) + self.beta * torch.mean((zq - z.detach())**2)

        zq = z + (zq - z).detach()

        e_mean = torch.mean(nearest, dim=0)
        perplexity = (-torch.sum(e_mean * torch.log(e_mean + 1e-10))).exp()

        zq = zq.permute(0, 3, 1, 2).contiguous()
        return zq, loss, perplexity, nearest, nearest_idx


class VQ_VAE(nn.Module):
    def __init__(self,
            n_channels=3,
            h_dim=128,
            res_dim=32,
            n_res_blocks=2,
            n_emb=512,
            emb_dim=64,
            beta=0.25,
            device=torch.device('cpu')):
        super().__init__()

        self.encoder = Encoder(n_channels, h_dim, n_res_blocks, res_dim)
        self.downsample = nn.Conv2d(h_dim, emb_dim, 1, 1)

        self.quantizer = VectorQuantizer(n_emb, emb_dim, beta, device)

        self.upsample = nn.ConvTranspose2d(h_dim, h_dim, 3, 1, padding=1)
        self.decoder = Decoder(h_dim, n_channels, n_res_blocks, res_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = self.downsample(z)

        zq, emb_loss, perplexity, *_ = self.quantizer(z)

        zq = self.upsample(zq)
        recon_x = self.decoder(zq)
        return recon_x, emb_loss, perplexity

        
if __name__ == '__main__':
    imgs = torch.rand(128, 3, 64, 64).cuda()
    model = VQ_VAE(device='cuda').cuda()
    out = model(imgs)

    print(*out)

