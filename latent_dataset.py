import torch
import torch.nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torchvision import transforms as T
from torchvision import datasets

from vqvae import VQ_VAE

torch.backends.cudnn.benchmark=True




if __name__ == '__main__':
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

    weight = accelerator.load('log/latest.pth')
    accelerator.unwrap_model(model).load_state_dict(weight)
    model.eval()

    emb_idx = []
    size = args.img_size // 4

    for img, _ in dataloader:
        img = img.to(device)

        idx = model.get_codebook(img)
        idx = idx.reshape(img.shape[0], size, size)
        emb_idx.append(idx.detach().clone())

    emb_idx = accelerator.gather(emb_idx)
    emb_idx = torch.cat(emb_idx)
    print(emb_idx.shape)
    torch.save(emb_idx, 'latent_data.pth')

