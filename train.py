import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from torchvision import utils, datasets

from model import VQ_VAE



torch.backends.cudnn.benchmark = True



def main(args):
    device = 'cuda'

    transform = TF.Compose([
        TF.Resize(args.img_size),
        TF.CenterCrop(args.img_size),
        TF.ToTensor(),
        TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    dataloader = DataLoader(
        datasets.ImageFolder(args.path, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model = VQ_VAE(device=device).to(device)

    recon_loss = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0

        for img, _ in dataloader:
            img = img.to(device)

            out, emb_loss = model(img)

            loss = recon_loss(img, out) + emb_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach() * img.shape[0]

        print(f'Epoch:{e}, Loss: {l/n}({l}/{n})'.format(
            e=epoch, l=train_loss, n=len(dataloader.dataset)))

        if i % args.log_interval == 0:
            utils.save_image(
                torch.cat([img, out], 0),
                f'log/recon{str(epoch+1).zfill(4)}.png',
                range=(-1, 1)
            )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--log-interval', type=int, default=5)

    os.makedirs('log', exist_ok=True)
    
    main(parser.parse_args())

