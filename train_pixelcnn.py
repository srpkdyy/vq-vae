import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms as TF
from torchvision import utils, datasets
from accelerate import Accelerator

from vqvae import VQ_VAE
from pixelcnn import PixelCNN



torch.backends.cudnn.benchmark = True



def main(args):
    accelerator = Accelerator()
    device = accelerator.device

    ds = TensorDataset(torch.load('latent_data.pth'))

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

    model = PixelCNN().to(device)
    if accelerator.is_local_main_process:
        vq_vae = VQ_VAE(device=device).to(device)
        weights = torch.load('log/latest.pth')
        accelerator.unwrap_model(vq_vae).load_state_dict(weights)
        vq_vae.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataloader, model, optimizer = accelerator.prepare(
        dataloader, model, optimizer
    )

    accelerator.print('Start training...')
    for epoch in range(args.epoch):
        model.train()

        n = 0
        train_loss = 0

        for x in dataloader:
            x = x[0].to(device)
            target = x

            out = model(x)

            loss = criterion(out, target)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            n += x.shape[0]
            train_loss += loss.detach() * x.shape[0]

        train_loss /= n
        train_loss = accelerator.gather(train_loss)

        accelerator.print(f'Epoch:{epoch+1}, Loss:{train_loss.sum().float()}')

        if accelerator.is_local_main_process and epoch % args.log_interval == 0:
            sample = accelerator.unwrap_model(model).sample(args.img_size, 16)

            imgs = vq_vae.decode_code(sample)

            utils.save_image(
                imgs,
                f'log/generate{str(epoch+1).zfill(4)}.png',
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
    parser.add_argument('-b', '--batch-size', type=int, default=1024)
    parser.add_argument('-e', '--epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img-size', type=int, default=16)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--log-dir', type=str, default='log')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)

