import math

from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
import shutil


def split_into_blocks(image, block_sz):
    blocks = []
    for i in range(0, image.shape[0], block_sz):
        for j in range(0, image.shape[1], block_sz):
            blocks.append(image[i:i + block_sz, j:j + block_sz])  # first row, then column
    return np.array(blocks)

def combine_blocks(blocks, height, width, block_sz):
    image = np.zeros((height, width), np.float32)
    index = 0
    for i in range(0, height, block_sz):
        for j in range(0, width, block_sz):
            image[i:i + block_sz, j:j + block_sz] = blocks[index]
            index += 1
    return image

def dct_transform(blocks):
    dct_blocks = []
    for block in blocks:
        dct_block = np.float32(block) - 128  # Shift to center around 0
        dct_block = cv2.dct(dct_block)
        dct_blocks.append(dct_block)
    return np.array(dct_blocks)

def idct_transform(blocks):
    idct_blocks = []
    for block in blocks:
        idct_block = cv2.idct(block)
        idct_block = idct_block + 128  # Shift back
        idct_blocks.append(idct_block)
    return np.array(idct_blocks)



def DCT_upsampling(img_array=None, block_sz=None):
    assert block_sz == 4
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]

    img_height = img_array.shape[0]
    img_width = img_array.shape[1]

    img_y = 0.299 * R + 0.587 * G + 0.114 * B
    img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

    y_blocks = split_into_blocks(img_y, block_sz)  # Y component, (blocks, 16) --> (blocks, 4, 4)
    dct_y_blocks = dct_transform(y_blocks)  # (blocks, 4, 4)
    num_blocks = dct_y_blocks.shape[0]
    up_dct_y = np.zeros((num_blocks, block_sz * 2, block_sz * 2))
    for n in range(num_blocks):
        for i in range(block_sz):
            for j in range(block_sz):
                up_dct_y[n, i, j] = dct_y_blocks[n, i, j] / (
                            math.cos(math.pi * i / 16) * math.cos(math.pi * j / 16) * 0.5)

    cb_blocks = split_into_blocks(img_cb, block_sz)  # Cb component, (blocks, 16) --> (blocks, 4, 4)
    dct_cb_blocks = dct_transform(cb_blocks)  # (blocks, 4, 4)
    num_blocks = dct_cb_blocks.shape[0]
    up_dct_cb = np.zeros((num_blocks, block_sz * 2, block_sz * 2))
    for n in range(num_blocks):
        for i in range(block_sz):
            for j in range(block_sz):
                up_dct_cb[n, i, j] = dct_cb_blocks[n, i, j] / (
                            math.cos(math.pi * i / 16) * math.cos(math.pi * j / 16) * 0.5)

    cr_blocks = split_into_blocks(img_cr, block_sz)  # Cr component, (blocks, 16) --> (blocks, 4, 4)
    dct_cr_blocks = dct_transform(cr_blocks)  # (blocks, 4, 4)
    num_blocks = dct_cr_blocks.shape[0]
    up_dct_cr = np.zeros((num_blocks, block_sz * 2, block_sz * 2))
    for n in range(num_blocks):
        for i in range(block_sz):
            for j in range(block_sz):
                up_dct_cr[n, i, j] = dct_cr_blocks[n, i, j] / (
                            math.cos(math.pi * i / 16) * math.cos(math.pi * j / 16) * 0.5)

    # Apply Inverse DCT on each block
    idct_y_blocks = idct_transform(up_dct_y)
    idct_cb_blocks = idct_transform(up_dct_cb)
    idct_cr_blocks = idct_transform(up_dct_cr)

    y_reconstructed = combine_blocks(idct_y_blocks, img_height * 2, img_width * 2, block_sz * 2)
    cb_reconstructed = combine_blocks(idct_cb_blocks, img_height * 2, img_width * 2, block_sz * 2)
    cr_reconstructed = combine_blocks(idct_cr_blocks, img_height * 2, img_width * 2, block_sz * 2)

    R = y_reconstructed + 1.402 * (cr_reconstructed - 128)
    G = y_reconstructed - 0.344136 * (cb_reconstructed - 128) - 0.714136 * (cr_reconstructed - 128)
    B = y_reconstructed + 1.772 * (cb_reconstructed - 128)

    rgb_reconstructed = np.zeros((img_height * 2, img_width * 2, 3))
    rgb_reconstructed[:, :, 0] = np.clip(R, 0, 255)
    rgb_reconstructed[:, :, 1] = np.clip(G, 0, 255)
    rgb_reconstructed[:, :, 2] = np.clip(B, 0, 255)

    return np.uint8(rgb_reconstructed)  # (h, w, 3), RGB channels


def upsampling_comparison(img_path=None, block_sz=4, downsampled_sz=128):
    image = Image.open(img_path)
    image.save(f"DCTfs_org_image.png")

    image = image.resize((downsampled_sz, downsampled_sz), Image.BICUBIC)
    print(f"downsampled image size: {downsampled_sz}x{downsampled_sz}")
    upsampled_image = image.resize((downsampled_sz * 2, downsampled_sz * 2), Image.BICUBIC)

    image.save(f"DCTfs_downsample_image.png")
    upsampled_image.save(f"DCTfs_upsample_image.png")

    img = np.array(image)
    umsampled_img = DCT_upsampling(img_array=img, block_sz=block_sz)
    dct_upsample_image = Image.fromarray(umsampled_img)
    dct_upsample_image.save('DCTfs_DCTupsample_image.png')


def select_50k_images(main_folder, org_dataset=None):
    os.makedirs(org_dataset, exist_ok=True)
    all_files = [f for f in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, f))]
    selected_files = random.sample(all_files, 50000)  # Randomly select 50000 images

    for file in tqdm(selected_files):
        shutil.copy(os.path.join(main_folder, file), os.path.join(org_dataset, file))

def DCT_upsam_FID(org_dataset=None, upsam_dataset=None, dct_upsam_dataset=None):
    os.makedirs(upsam_dataset, exist_ok=True)

    if dct_upsam_dataset:
        os.makedirs(dct_upsam_dataset, exist_ok=True)

    all_files = [f for f in os.listdir(org_dataset) if os.path.isfile(os.path.join(org_dataset, f))]
    for file in tqdm(all_files):

        # Save the upsampled images
        img_path = os.path.join(org_dataset, file)
        img = Image.open(img_path)

        # FID: LANCZOS 12.973, NEAREST 8.157, BICUBIC 12.528, DCT_upsam 9.788
        img_downsampled = img.resize((img.width // 2, img.height // 2), Image.BICUBIC)  # Image.NEAREST, Image.BICUBIC
        img_upsampled = img_downsampled.resize((img_downsampled.width * 2, img_downsampled.height * 2), Image.BICUBIC)
        img_upsampled.save(os.path.join(upsam_dataset, file))

        if dct_upsam_dataset:
            img = np.array(img_downsampled)
            umsampled_img = DCT_upsampling(img_array=img, block_sz=4)
            dct_upsample_image = Image.fromarray(umsampled_img)
            dct_upsample_image.save(os.path.join(dct_upsam_dataset, file))


if __name__ == "__main__":
    upsampling_comparison(img_path='/home/mang/PycharmProjects/U-ViT/ajpeg-dif/jpeg-dif/black-swan.jpg')
    # select_50k_images('/home/mang/Downloads/ffhq256_jpg/ffhq256',
    #                   org_dataset='/home/mang/Downloads/ffhq256_50k')
    # DCT_upsam_FID(org_dataset='/home/mang/Downloads/ffhq256_50k',
    #               upsam_dataset='/home/mang/Downloads/ffhq256_50k_upsampled',
    #               dct_upsam_dataset='/home/mang/Downloads/ffhq256_50k_upsampled_dct')