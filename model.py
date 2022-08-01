import torch
import torch.nn


class Encoder(nn.Module):
    pass


class VectorQuantizer(nn.Module):
    def __init__(self, n_emb, emb_dim, commit_cost):
        super().__init__()
        self.emb = nn.Embedding(n_emb, emb_dim)


class Decoder(nn.Module):
    pass


class VQ_VAE(nn.Module):
    pass
        
