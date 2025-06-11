import os
import sys
project_path = os.path.dirname((os.path.dirname(__file__)))
sys.path.append(project_path)

import torch.nn as nn
import numpy as np
import torch
from datasets import FFHQ256
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
from tqdm import tqdm


torch.manual_seed(0)
np.random.seed(0)


def main(
        resolution=256,
        path=f'/home/mang/Downloads/ffhq256_jpg',
        folder=f'/home/mang/Downloads/ffhq256_latents/features'
        ):

    dataset = FFHQ256(path, resolution=resolution, random_flip=False)
    train_dataset = dataset.get_split(split='train', labeled=False)
    train_dataset_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    model = get_model('assets/stable-diffusion/autoencoder_kl.pth')
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    idx = 0
    if not os.path.exists(folder):
        os.makedirs(folder)

    for batch in tqdm(train_dataset_loader):
        img = batch
        img = torch.cat([img, img.flip(dims=[-1])], dim=0)  # flip image, (batch*2, 3, 256, 256)
        img = img.to(device)
        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()  # (batch*2, 4*2, 32, 32)

        for moment in moments:  # (8, 32, 32)
            np.save(f'{folder}/{idx}.npy', moment)
            idx += 1

    print(f'saved {idx} files')


if __name__ == "__main__":
    main()
