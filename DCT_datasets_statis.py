from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random


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


def image_to_DCT_array(dataset=None, img_folder=None, block_sz=8, coe=None):
    Y_coe = []
    Cb_coe = []
    Cr_coe = []
    file_list = os.listdir(img_folder)
    sampled_files = random.sample(file_list, 50000)

    for filename in tqdm(sampled_files):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            img_path = os.path.join(img_folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)

            # Step 1: Convert RGB to YCbCr
            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]

            img_y = 0.299 * R + 0.587 * G + 0.114 * B
            img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
            img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

            cb_downsampled = cv2.resize(img_cb, (img_cb.shape[1] // 2, img_cb.shape[0] // 2),
                                        interpolation=cv2.INTER_LINEAR)
            cr_downsampled = cv2.resize(img_cr, (img_cr.shape[1] // 2, img_cr.shape[0] // 2),
                                        interpolation=cv2.INTER_LINEAR)

            # Step 2: Split the Y, Cb, and Cr components into 8x8 blocks
            # Step 3: Apply DCT on each block
            if coe.lower() == 'y':
                y_blocks = split_into_blocks(img_y, block_sz)  # Y component, (64, 64) --> (64, 8, 8)
                dct_y_blocks = dct_transform(y_blocks)  # (64, 8, 8)
                dct_y_blocks = dct_y_blocks.astype(np.float16)
                Y_coe.append(dct_y_blocks.reshape(-1, block_sz*block_sz))
            elif coe.lower() == 'cb':
                cb_blocks = split_into_blocks(cb_downsampled, block_sz)  # Cb component, (32, 32) --> (16, 8, 8)
                dct_cb_blocks = dct_transform(cb_blocks)  # (16, 8, 8)
                dct_cb_blocks = dct_cb_blocks.astype(np.float16)
                Cb_coe.append(dct_cb_blocks.reshape(-1, block_sz*block_sz))
            elif coe.lower() == 'cr':
                cr_blocks = split_into_blocks(cr_downsampled, block_sz)  # Cr component, (32, 32) --> (16, 8, 8)
                dct_cr_blocks = dct_transform(cr_blocks)  # (16, 8, 8)
                dct_cr_blocks = dct_cr_blocks.astype(np.float16)
                Cr_coe.append(dct_cr_blocks.reshape(-1, block_sz*block_sz))

    if coe.lower() == 'y':
        Y_coe = np.array(Y_coe).reshape(-1, block_sz*block_sz)
        print(f"yield Y with shape {Y_coe.shape}")
        np.save(f'/home/mang/Downloads/{dataset}_{block_sz}by{block_sz}_y', Y_coe)
        print(f"y: {Y_coe.shape} saved")

    if coe.lower() == 'cb':
        Cb_coe = np.array(Cb_coe).reshape(-1, block_sz * block_sz)
        print(f"yield Cb with shape {Cb_coe.shape}")
        np.save(f'/home/mang/Downloads/{dataset}_{block_sz}by{block_sz}_cb', Cb_coe)
        print(f"cb: {Cb_coe.shape} saved")

    elif coe.lower() == 'cr':
        Cr_coe = np.array(Cr_coe).reshape(-1, block_sz * block_sz)
        print(f"yield Cr with shape {Cr_coe.shape}")
        np.save(f'/home/mang/Downloads/{dataset}_{block_sz}by{block_sz}_cr', Cr_coe)
        print(f"cr: {Cr_coe.shape} saved")


def DCT_statis_from_array(array_path=None, block_sz=None, tau=98.25, eta=None):
    coe_array = np.load(array_path)
    print(f"{coe_array.shape} loaded from {array_path}")

    """statistics of eta and entropy"""
    DCT_coe_bounds = []
    low_thresh = 100 - tau
    up_thresh = tau
    for index in range(block_sz * block_sz):
        data = coe_array[:, index].astype(np.float64)  # avoid overflow
        lower_bound = np.percentile(data, low_thresh)
        upper_bound = np.percentile(data, up_thresh)
        data = data[(data >= lower_bound) & (data <= upper_bound)]

        mean = np.around(np.mean(data), decimals=3)
        print(f"({up_thresh - low_thresh}%) coe {index} has upper bound {upper_bound} and lower bound {lower_bound}")

        if np.abs(upper_bound) > np.abs(lower_bound):
            upper_bound = np.around(np.abs(upper_bound), decimals=3)
            DCT_coe_bounds.append(upper_bound)
        else:
            lower_bound = np.around(np.abs(lower_bound), decimals=3)
            DCT_coe_bounds.append(np.abs(lower_bound))

    print(f"{up_thresh - low_thresh} percentile bound is {DCT_coe_bounds}:")
    print(f"eta is {DCT_coe_bounds[0]}")
    print(f"{'-' * 100}")

    # approximate the entropy by histogram
    if eta:
        entropys = []
        for i in range(block_sz ** 2):
            DCT_coe = coe_array[:, i].astype(np.float64)  # avoid overflow
            lower_bound = np.percentile(DCT_coe, low_thresh)
            upper_bound = np.percentile(DCT_coe, up_thresh)
            filtered_coe = DCT_coe[(DCT_coe > lower_bound) & (DCT_coe < upper_bound)]
            filtered_coe = filtered_coe / eta  # first get Y_bound, then compute entropy

            counts, bin_edges = np.histogram(filtered_coe, bins=100, range=(-1, 1))
            probabilities = counts / np.sum(counts)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small epsilon to avoid log(0)
            entropy = np.around(entropy, decimals=3)
            entropys.append(entropy)

        print(f"entropy: {entropys}")


def mask_high_freq_coe_from_img_folder(img_folder=None, save_folder=None, img_sz=256, block_sz=4, low_freqs=None, eta=None):

    # parameters of DCT transform
    Y = int(img_sz * img_sz / (block_sz * block_sz))  # num of Y blocks
    Y_blocks_per_row = int(img_sz / block_sz)
    cb_blocks_per_row = int((img_sz / block_sz) / 2)
    index = []  # index of Y if merging 2*2 Y-block area
    for row in range(0, Y, int(2 * Y_blocks_per_row)):  # 0, 32, 64...
        for col in range(0, Y_blocks_per_row, 2):  # 0, 2, 4...
            index.append(row + col)
    assert len(index) == int(Y / 4)

    tokens = int((img_sz / (block_sz * 2))**2)
    print(f"num of tokens: {tokens}")
    num_y_blocks = tokens * 4
    num_cb_blocks = tokens

    if block_sz == 4:
        low2high_order = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
        reverse_order = [0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15]
    elif block_sz == 8:
        low2high_order = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
                          27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                          58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
        reverse_order = [0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
                         18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37,
                         47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63]
    else:
        raise ValueError (f"{block_sz} not implemented")

    # token sequence: 4Y-Cb-Cr-4Y-Cb-Cr...
    cb_index = [i for i in range(4, tokens, 6)]
    cr_index = [i for i in range(5, tokens, 6)]
    y_index = [i for i in range(0, tokens) if i not in cb_index and i not in cr_index]
    assert len(y_index) + len(cb_index) + len(cr_index) == tokens

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"folder {save_folder} created")

    cnt = 0
    for filename in tqdm(os.listdir(img_folder)):
        cnt += 1
        if cnt <= 50000:
            """forward DCT"""
            img_path = os.path.join(img_folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)

            # Step 1: Convert RGB to YCbCr
            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]

            img_y = 0.299 * R + 0.587 * G + 0.114 * B
            img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
            img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

            cb_downsampled = cv2.resize(img_cb, (img_cb.shape[1] // 2, img_cb.shape[0] // 2),
                                        interpolation=cv2.INTER_LINEAR)
            cr_downsampled = cv2.resize(img_cr, (img_cr.shape[1] // 2, img_cr.shape[0] // 2),
                                        interpolation=cv2.INTER_LINEAR)

            # Step 2: Split the Y, Cb, and Cr components into 4x4 blocks
            y_blocks = split_into_blocks(img_y, block_sz)  # Y component, (64, 64) --> (256, 4, 4)
            cb_blocks = split_into_blocks(cb_downsampled, block_sz)  # Cb component, (32, 32) --> (64, 4, 4)
            cr_blocks = split_into_blocks(cr_downsampled, block_sz)  # Cr component, (32, 32) --> (64, 4, 4)

            # Step 3: Apply DCT on each block
            dct_y_blocks = dct_transform(y_blocks)  # (256, 4, 4)
            dct_cb_blocks = dct_transform(cb_blocks)  # (64, 4, 4)
            dct_cr_blocks = dct_transform(cr_blocks)  # (64, 4, 4)

            # Step 4: organize the token order by Y-Y-Y-Y-Cb-Cr (2_blocks*2_blocks pixel region)
            DCT_blocks = []
            for i in range(dct_cr_blocks.shape[0]):
                DCT_blocks.append([
                    dct_y_blocks[index[i]],  # Y
                    dct_y_blocks[index[i] + 1],  # Y
                    dct_y_blocks[index[i] + Y_blocks_per_row],  # Y
                    dct_y_blocks[index[i] + Y_blocks_per_row + 1],  # Y
                    dct_cb_blocks[i],  # Cb
                    dct_cr_blocks[i],  # Cr
                ])
            DCT_blocks = np.array(DCT_blocks).reshape(-1, 6, block_sz * block_sz)  # (64, 6, 4, 4) --> (64, 6, 16)

            # Step 5: scale into [-1, 1]
            assert DCT_blocks.shape == (tokens, 6, block_sz * block_sz)
            DCT_blocks[:, :4 :] = (DCT_blocks[:, :4 :] ) / eta
            DCT_blocks[:, 4, :] = (DCT_blocks[:, 4, :] ) / eta
            DCT_blocks[:, 5, :] = (DCT_blocks[:, 5, :] ) / eta

            # Step 6: reorder coe from low to high freq, then mask out high-freq signals
            DCT_blocks = DCT_blocks[:, :, low2high_order]  # (64, 6, 16) --> (64, 6, 16)
            DCT_blocks = DCT_blocks[:, :, :low_freqs]  # (64, 6, 16) --> (64, 6, low_freq_coe)

            """Inverse DCT"""
            DCT = np.zeros((tokens, 6, block_sz * block_sz))  # (tokens, 6, 16)
            DCT[:, :, :low_freqs] = DCT_blocks
            DCT = DCT[..., reverse_order]  # convert the low to high freq order back to 8*8 order

            DCT_Y = (DCT[:, :4, :] * eta)  # (64, 4, 16)
            DCT_Cb = (DCT[:, 4, :] * eta)  # (64, 16)
            DCT_Cr = (DCT[:, 5, :] * eta)  # (64, 16)

            DCT_Cb = DCT_Cb.reshape(num_cb_blocks, block_sz, block_sz)  # (64, 16) --> (64, 4, 4)
            DCT_Cr = DCT_Cr.reshape(num_cb_blocks, block_sz, block_sz)  # (64, 16) --> (64, 4, 4)

            y_blocks = []
            for row in range(cb_blocks_per_row):  # 16 cb/cr blocks, so 4*4 spatial blocks
                tem_ls = []
                for col in range(cb_blocks_per_row):
                    ind = row * cb_blocks_per_row + col
                    y_blocks.append(DCT_Y[ind, 0, :])
                    y_blocks.append(DCT_Y[ind, 1, :])
                    tem_ls.append(DCT_Y[ind, 2, :])
                    tem_ls.append(DCT_Y[ind, 3, :])
                for ele in tem_ls:
                    y_blocks.append(ele)
            DCT_Y = np.array(y_blocks).reshape(num_y_blocks, block_sz, block_sz)  # (256, 4, 4)

            # Apply Inverse DCT on each block
            idct_y_blocks = idct_transform(DCT_Y)
            idct_cb_blocks = idct_transform(DCT_Cb)
            idct_cr_blocks = idct_transform(DCT_Cr)

            # Combine blocks back into images
            height, width = img_sz, img_sz
            y_reconstructed = combine_blocks(idct_y_blocks, height, width, block_sz)
            cb_reconstructed = combine_blocks(idct_cb_blocks, int(height / 2), int(width / 2), block_sz)
            cr_reconstructed = combine_blocks(idct_cr_blocks, int(height / 2), int(width / 2), block_sz)

            # Upsample Cb and Cr to original size
            cb_upsampled = cv2.resize(cb_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)
            cr_upsampled = cv2.resize(cr_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)

            # Step 5: Convert YCbCr back to RGB
            R = y_reconstructed + 1.402 * (cr_upsampled - 128)
            G = y_reconstructed - 0.344136 * (cb_upsampled - 128) - 0.714136 * (cr_upsampled - 128)
            B = y_reconstructed + 1.772 * (cb_upsampled - 128)

            rgb_reconstructed = np.zeros((height, width, 3))
            rgb_reconstructed[:, :, 0] = np.clip(R, 0, 255)
            rgb_reconstructed[:, :, 1] = np.clip(G, 0, 255)
            rgb_reconstructed[:, :, 2] = np.clip(B, 0, 255)

            # Convert to uint8 and save img
            rgb_reconstructed = np.uint8(rgb_reconstructed)  # (h, w, 3), RGB channels
            rgb_reconstructed = Image.fromarray(rgb_reconstructed)
            save_path = os.path.join(save_folder, filename)
            rgb_reconstructed.save(save_path)

        else:
            break


if __name__ == "__main__":

    """cifar10"""
    # draw 50k samples from the RGB dataset, convert then into DCT arrays for later DCT statis
    image_to_DCT_array(dataset='cifar10', img_folder='/home/mang/Downloads/cifar10/cifar_train', block_sz=2, coe='y')

    # get eta
    DCT_statis_from_array(array_path='/home/mang/Downloads/cifar10_2by2_y.npy', block_sz=2, tau=98.25)

    # get entropy for Entropy-Based Frequency Reweighting (EBFR)
    DCT_statis_from_array(array_path='/home/mang/Downloads/cifar10_2by2_y.npy', block_sz=2, tau=98.25, eta=242.382)

    """celeba 64"""
    image_to_DCT_array(dataset='celeba64', img_folder='/home/mang/Downloads/celeba/celeba64', block_sz=2, coe='y')
    DCT_statis_from_array(array_path='/home/mang/Downloads/celeba64_2by2_y.npy',
                          block_sz=2, tau=98.25)
    DCT_statis_from_array(array_path='/home/mang/Downloads/celeba64_2by2_y.npy',
                          block_sz=2, tau=98.25, eta=244.925)

    """imagenet 64"""
    image_to_DCT_array(dataset='imagenet64', img_folder='/home/mang/Downloads/imagenet64/train', block_sz=2, coe='y')
    DCT_statis_from_array(array_path='/home/mang/Downloads/imagenet64_2by2_y.npy',
                          block_sz=2, tau=98.25)
    DCT_statis_from_array(array_path='/home/mang/Downloads/imagenet64_2by2_y.npy',
                          block_sz=2, tau=98.25, eta=247.125)

    """ffhq 128"""
    image_to_DCT_array(dataset='ffhq128', img_folder='/home/mang/Downloads/ffhq128/ffhq128', block_sz=4, coe='y')
    DCT_statis_from_array(array_path='/home/mang/Downloads/ffhq128_4by4_y.npy',
                          block_sz=4, tau=98.25)
    DCT_statis_from_array(array_path='/home/mang/Downloads/ffhq128_4by4_y.npy',
                          block_sz=4, tau=98.25, eta=480.25)

    """FFHQ 256"""
    image_to_DCT_array(dataset='ffhq256', img_folder='/home/mang/Downloads/ffhq256/ffhq256', block_sz=4, coe='y')
    DCT_statis_from_array(array_path='/home/mang/Downloads/ffhq256_4by4_y.npy',
                          block_sz=4, tau=98.25)
    DCT_statis_from_array(array_path='/home/mang/Downloads/ffhq256_4by4_y.npy',
                          block_sz=4, tau=98.25, eta=485.0)

    """decide m* (Eq. (6) in the paper)"""
    # get P(dct_data(m)), where m = B*B - low_freqs
    # then compute FID(P_data, P(dct_data(m))) using repo https://github.com/mseitzer/pytorch-fid
    mask_high_freq_coe_from_img_folder(img_folder='/home/mang/Downloads/ffhq256/ffhq256',
                                       save_folder='/home/mang/Downloads/recon_ffhq256_coe8',
                                       img_sz=256, block_sz=4, low_freqs=8)

