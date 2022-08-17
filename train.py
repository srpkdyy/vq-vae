import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as TF
from torchvision import utils, datasets
from accelerate import Accelerator

from model import VQ_VAE



torch.backends.cudnn.benchmark = True



def main(args):
    accelerator = Accelerator()
    device = accelerator.device

    transform = TF.Compose([
        TF.Resize(args.img_size),
        TF.CenterCrop(args.img_size),
        TF.ToTensor(),
        TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    ds = datasets.STL10(
        args.path,
        split='unlabeled',
        transform=transform
    )
    dataloader = DataLoader(
        ds,
        #datasets.ImageFolder(args.path, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    accelerator.print(f'Dataset length: {len(dataloader.dataset)}')

    model = VQ_VAE(device=device).to(device)

    recon_loss = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataloader, model, optimizer = accelerator.prepare(
        dataloader, model, optimizer
    )

    for epoch in range(args.epoch):
        model.train()

        n = 0
        train_loss = 0

        for img, _ in dataloader:
            img = img.to(device)

            out, emb_loss = model(img)

            loss = recon_loss(img, out) + emb_loss

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            n += img.shape[0]
            train_loss += loss.detach() * img.shape[0]

        train_loss /= n
        train_loss = accelerator.gather(train_loss)

        accelerator.print(f'Epoch:{epoch+1}, Loss:{train_loss.sum().float()}')

        if accelerator.is_local_main_process and epoch % args.log_interval == 0:
            utils.save_image(
                torch.cat([img[:8], out[:8]], 0),
                f'log/recon{str(epoch+1).zfill(4)}.png',
                normalize=True,
                value_range=(-1, 1)
            )

    accelerator.wait_for_everyone()
    accelerator.save(
        accelerator.unwrap_model(model).state_dict(),
        os.path.join(args.log_dir, 'latest.pth')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--log-dir', type=str, default='log')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)

