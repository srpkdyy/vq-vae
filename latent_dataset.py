import torch
import torch.nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torchvision import transforms as T
from torchvision import datasets
from tqdm import tqdm

from vqvae import VQ_VAE

torch.backends.cudnn.benchmark=True




def save_latent(args):
    accelerator = Accelerator()
    device = accelerator.device

    transform = T.Compose([
        T.Resize(args.img_size),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ds = datasets.STL10(
        args.path,
        split='unlabeled',
        transform=transform
    )

    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    model = VQ_VAE(device=device).to(device)

    dataloader, model = accelerator.prepare(
        dataloader, model
    )

    weight = torch.load('log/latest.pth')
    accelerator.unwrap_model(model).load_state_dict(weight)
    model.eval()

    emb_idx = []
    size = args.img_size // 4

    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)

    for img, _ in pbar:
        img = img.to(device)

        idx = model.get_codebook(img)
        idx = idx.reshape(img.shape[0], size, size)
        emb_idx.append(idx.detach().clone())

    emb_idx = accelerator.gather(emb_idx)
    emb_idx = torch.cat(emb_idx)
    print(emb_idx.shape)
    torch.save(emb_idx, 'latent_data.pth')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-b', '--batch-size', type=int, default=1024)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--log-dir', type=str, default='log')
    args = parser.parse_args()

    save_latent(args)

