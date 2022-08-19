import torch
import torch.nn as nn
import torch.nn.functioinal as F


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        h, w = self.weight.size()[2:4]
        self.mask.fill_(1)

        self.mask[:, :, h//2 + 1:] = 0
        if mask_type == 'A':
            self.mask[:, :, h//2, w//2:] = 0
        else:
            self.mask[:, :, h//2, w//2 + 1:] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):
    def __init__(self, n_hidden=64, n_out=512, n_layers=7):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(MaskedConv2d(mask_type='A',
                                        in_channels=1,
                                        out_channels=n_hidden,
                                        kernel_size=7,
                                        stride=1,
                                        padding=3,
                                        bias=False))
        self.layers.append(nn.BatchNorm2d(n_hidden))
        self.layers.append(nn.ReLU(True))

        for _ in range(n_layers):
            self.layers.append(MaskedConv2d(mask_type='B',
                                            in_channels=n_hidden,
                                            out_channels=n_hidden,
                                            kernel_size=7,
                                            stride=1,
                                            padding=3,
                                            bias=False))
            self.layers.append(nn.BatchNorm2d(n_hidden))
            self.layers.append(nn.ReLU(True))

        self.layers.append(nn.Conv2d(n_hidden, n_out, 1))

        self.apply(init_weights)


    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def sample(self, size, n_samples):
        device = next(self.parameters()).device
        x = torch.zeros(
            (n_samples, size, size),
            dtype=torch.int64,
            device=device)

        for i in range(size):
            for j in range(size):
                out = self.forward(x)
                probs = F.softmax(out[:, :, i, j], 1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )

        return x



