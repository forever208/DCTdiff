from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

#########################
##### block size = 2 ####
#########################

# before reorder
CELEBA64_Y_bound = np.array(
    [246.354, 48.555, 43.641, 15.701]
)

CELEBA64_Cb_bound = np.array(
    [76.736, 7.473, 5.911, 1.189]
)

CELEBA64_Cr_bound = np.array(
    [91.474, 8.668, 7.153, 1.302]
)

# bound of 96.5% interval without mean subtraction
# CELEBA64_Y_bound = np.array(
#     [244.925, 244.925, 244.925, 244.925]
# )
#
# CELEBA64_Cb_bound = np.array(
#     [244.925, 244.925, 244.925, 244.925]
# )
#
# CELEBA64_Cr_bound = np.array(
#     [244.925, 244.925, 244.925, 244.925]
# )


# std before normalization
Y_std = np.array(
    [77.953, 15.432, 13.612, 5.06]
)

Cb_std = np.array(
    [21.924, 2.8, 2.086, 0.468]
)

Cr_std = np.array(
    [26.072, 3.28, 2.514, 0.516]
)


def celeba_to_celeba64x64(input_folder=None, output_folder=None):
    print(f"found {len(os.listdir(input_folder))} images in {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    cx = 89
    cy = 121
    x1 = cx - 64
    x2 = cx + 64
    y1 = cy - 64
    y2 = cy + 64

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # center crop and resize to 64x64
            with Image.open(input_path) as img:
                img = img.crop((x1, y1, x2, y2)).resize((64, 64))
                img.save(output_path)

    print(f"generated {len(os.listdir(output_folder))} images in {output_folder}")


def DCT_and_InverDCT(img_path, de_DCT=False):
    def split_into_blocks(image):
        blocks = []
        for i in range(0, image.shape[0], 8):
            for j in range(0, image.shape[1], 8):
                blocks.append(image[i:i + 8, j:j + 8])  # first row, then column
        return np.array(blocks)

    def combine_blocks(blocks, height, width):
        image = np.zeros((height, width), np.float32)
        index = 0
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                image[i:i + 8, j:j + 8] = blocks[index]
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

    def quantization(dct_blocks, q_table):
        quantized_blocks = []
        for dct_block in dct_blocks:
            # quantized_block = np.round(dct_block / q_table)
            quantized_block = dct_block / q_table
            quantized_blocks.append(quantized_block)
        return np.array(quantized_blocks)

    def dequantization(dct_blocks, q_table):
        dequantized_blocks = []
        for dct_block in dct_blocks:
            dequantized_block = dct_block * q_table
            dequantized_blocks.append(dequantized_block)
        return np.array(dequantized_blocks)

    # Load the RGB image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('recon_original.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"original image saved into recon_original.jpg")

    # Step 1: Convert RGB to YCbCr
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    img_y = 0.299 * R + 0.587 * G + 0.114 * B
    img_cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    img_cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

    cb_downsampled = cv2.resize(img_cb, (img_cb.shape[1] // 2, img_cb.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    cr_downsampled = cv2.resize(img_cr, (img_cr.shape[1] // 2, img_cr.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    # Step 2: Split the Y, Cb, and Cr components into 8x8 blocks
    y_blocks = split_into_blocks(img_y)  # Y component, (256, 256) --> (1024, 8, 8)
    cb_blocks = split_into_blocks(cb_downsampled)  # Cb component, (256, 256) --> (1024, 8, 8)
    cr_blocks = split_into_blocks(cr_downsampled)  # Cr component, (256, 256) --> (1024, 8, 8)

    # Step 3: Apply Discrete Cosine Transform (DCT) on each block for Y, Cb, and Cr components
    dct_y_blocks = dct_transform(y_blocks)  # (1024, 8, 8)
    dct_cb_blocks = dct_transform(cb_blocks)  # (1024, 8, 8)
    dct_cr_blocks = dct_transform(cr_blocks)  # (1024, 8, 8)

    # # Step 4: use Quantization Tables (can be adjusted for different quality levels)
    q_table_y = np.array([[8, 6, 5, 8, 12, 20, 26, 31],
                          [6, 6, 7, 10, 13, 29, 30, 28],
                          [7, 7, 8, 12, 20, 29, 35, 28],
                          [7, 9, 11, 15, 26, 44, 40, 31],
                          [9, 11, 19, 28, 34, 55, 52, 39],
                          [12, 18, 28, 32, 41, 52, 57, 46],
                          [25, 32, 39, 44, 52, 61, 60, 51],
                          [36, 46, 48, 49, 56, 50, 52, 50]])

    q_table_cbcr = np.array([[8, 9, 12, 24, 50, 50, 50, 50],
                             [9, 11, 13, 33, 50, 50, 50, 50],
                             [12, 13, 28, 50, 50, 50, 50, 50],
                             [24, 33, 50, 50, 50, 50, 50, 50],
                             [50, 50, 50, 50, 50, 50, 50, 50],
                             [50, 50, 50, 50, 50, 50, 50, 50],
                             [50, 50, 50, 50, 50, 50, 50, 50],
                             [50, 50, 50, 50, 50, 50, 50, 50]])

    quantized_y_blocks = quantization(dct_y_blocks, q_table_y)
    quantized_cb_blocks = quantization(dct_cb_blocks, q_table_cbcr)
    quantized_cr_blocks = quantization(dct_cr_blocks, q_table_cbcr)

    np.set_printoptions(suppress=True, precision=3)
    # print(quantized_cr_blocks[1])

    # # mask out high frequency
    # mask = np.fromfunction(lambda i, j: i + j > 4, (8, 8), dtype=int)
    # quantized_y_blocks[:, mask] = 0
    # quantized_cb_blocks[:, mask] = 0
    # quantized_cr_blocks[:, mask] = 0

    import jpeglib
    im = jpeglib.read_dct(img_path)
    im.Y = quantized_y_blocks
    im.Cb = quantized_cb_blocks
    im.Cr = quantized_cr_blocks
    im.write_dct('recon_jpeglib_manual.jpg')
    print(f"manually DCT and jpeglib inverse DCT, save recon image into recon_jpeglib_manual.jpg")

    # manual inverse DCT
    if de_DCT:
        # Dequantize the DCT coefficients
        dequantized_y_blocks = dequantization(quantized_y_blocks, q_table_y)
        dequantized_cb_blocks = dequantization(quantized_cb_blocks, q_table_cbcr)
        dequantized_cr_blocks = dequantization(quantized_cr_blocks, q_table_cbcr)

        # Apply Inverse DCT on each block
        idct_y_blocks = idct_transform(dequantized_y_blocks)
        idct_cb_blocks = idct_transform(dequantized_cb_blocks)
        idct_cr_blocks = idct_transform(dequantized_cr_blocks)

        # Combine blocks back into images
        height, width = img_y.shape
        y_reconstructed = combine_blocks(idct_y_blocks, height, width)
        cb_reconstructed = combine_blocks(idct_cb_blocks, cb_downsampled.shape[0], cb_downsampled.shape[1])
        cr_reconstructed = combine_blocks(idct_cr_blocks, cr_downsampled.shape[0], cr_downsampled.shape[1])

        # Upsample Cb and Cr to original size
        cb_upsampled = cv2.resize(cb_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)
        cr_upsampled = cv2.resize(cr_reconstructed, (width, height), interpolation=cv2.INTER_LINEAR)

        # Step 5: Convert YCbCr back to RGB
        R = y_reconstructed + 1.402 * (cr_upsampled - 128)
        G = y_reconstructed - 0.344136 * (cb_upsampled - 128) - 0.714136 * (cr_upsampled - 128)
        B = y_reconstructed + 1.772 * (cb_upsampled - 128)

        rgb_reconstructed = np.zeros_like(image)
        rgb_reconstructed[:, :, 0] = np.clip(R, 0, 255)
        rgb_reconstructed[:, :, 1] = np.clip(G, 0, 255)
        rgb_reconstructed[:, :, 2] = np.clip(B, 0, 255)

        # Convert to uint8
        rgb_reconstructed = np.uint8(rgb_reconstructed)

        # Save or display the reconstructed image
        cv2.imwrite('recon_pure_manual.jpg', cv2.cvtColor(rgb_reconstructed, cv2.COLOR_RGB2BGR))
        print(f"manually DCT and inverse DCT implementation, save recon image into recon_pure_manual.jpg")


def images_to_array(image_folder=None, save_path=None):
    image_list = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if needed
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            image_array = np.array(image)
            image_list.append(image_array)

    combined_array = np.array(image_list)
    print(f"array shape: {combined_array.shape}")

    np.save(save_path, combined_array)
    print(f"array saved into {save_path}")


def image_to_DCT_array(img_folder=None, block_sz=8):
    def split_into_blocks(image, block_sz):
        blocks = []
        for i in range(0, image.shape[0], block_sz):
            for j in range(0, image.shape[1], block_sz):
                blocks.append(image[i:i + block_sz, j:j + block_sz])  # first row, then column
        return np.array(blocks)

    def dct_transform(blocks):
        dct_blocks = []
        for block in blocks:
            dct_block = np.float32(block) - 128  # Shift to center around 0
            dct_block = cv2.dct(dct_block)
            dct_blocks.append(dct_block)
        return np.array(dct_blocks)

    # read each image, do DCT transformation, compute the statistics
    Y_coe = []
    Cb_coe = []
    Cr_coe = []
    for filename in tqdm(os.listdir(img_folder)):
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
            y_blocks = split_into_blocks(img_y, block_sz)  # Y component, (64, 64) --> (64, 8, 8)
            cb_blocks = split_into_blocks(cb_downsampled, block_sz)  # Cb component, (32, 32) --> (16, 8, 8)
            cr_blocks = split_into_blocks(cr_downsampled, block_sz)  # Cr component, (32, 32) --> (16, 8, 8)

            # Step 3: Apply DCT on each block
            dct_y_blocks = dct_transform(y_blocks)  # (64, 8, 8)
            dct_cb_blocks = dct_transform(cb_blocks)  # (16, 8, 8)
            dct_cr_blocks = dct_transform(cr_blocks)  # (16, 8, 8)

            Y_coe.append(dct_y_blocks.reshape(-1, block_sz*block_sz))
            Cb_coe.append(dct_cb_blocks.reshape(-1, block_sz*block_sz))
            Cr_coe.append(dct_cr_blocks.reshape(-1, block_sz*block_sz))

    Y_coe = np.array(Y_coe).reshape(-1, block_sz*block_sz)
    Cb_coe = np.array(Cb_coe).reshape(-1, block_sz * block_sz)
    Cr_coe = np.array(Cr_coe).reshape(-1, block_sz * block_sz)
    print(f"yield Y with shape {Y_coe.shape}")
    print(f"yield Cb with shape {Cb_coe.shape}")
    print(f"yield Cr with shape {Cr_coe.shape}")

    np.save('/home/mang/Downloads/celeba64_2by2_y', Y_coe)
    print(f"y: {Y_coe.shape} saved")
    time.sleep(10)
    np.save('/home/mang/Downloads/celeba64_2by2_cb', Cb_coe)
    print(f"cb: {Cb_coe.shape} saved")
    time.sleep(10)
    np.save('/home/mang/Downloads/celeba64_2by2_cr', Cr_coe)
    print(f"cr: {Cr_coe.shape} saved")
    time.sleep(10)


def DCT_statis_from_array(array_path=None, block_sz=2):
    coe_array = np.load(array_path)  # input array is 8*8 coe block
    print(f"{coe_array.shape} loaded")

    """statistics for percentile, mean and std"""
    means = np.around(np.mean(coe_array, axis=0), decimals=3)
    stds = np.around(np.std(coe_array, axis=0), decimals=3)
    min = np.around(np.min(coe_array, axis=0), decimals=3)
    max = np.around(np.max(coe_array, axis=0), decimals=3)
    print(f"mean for {block_sz*block_sz} coe: {list(means)}")
    print(f"std for {block_sz*block_sz} coe: {list(stds)}")
    print(f"min for {block_sz*block_sz} coe: {list(min)}")
    print(f"max for {block_sz*block_sz} coe: {list(max)}")

    DCT_coe_bounds = []
    for index in range(block_sz * block_sz):
        low_thresh = 1.5
        up_thresh = 100 - low_thresh
        lower_bound = np.percentile(coe_array[:, index], low_thresh)
        upper_bound = np.percentile(coe_array[:, index], up_thresh)
        lower_bound = np.round(lower_bound, 3)
        upper_bound = np.round(upper_bound, 3)
        print(f"({up_thresh-low_thresh}%) coe {index} has upper bound {upper_bound} and lower bound {lower_bound} after subtracting mean {means[index]}")

        if np.abs(lower_bound) > np.abs(upper_bound):
            DCT_coe_bounds.append(np.abs(lower_bound))
        else:
            DCT_coe_bounds.append(np.abs(upper_bound))
    print(f"{up_thresh-low_thresh} percentile bound is {DCT_coe_bounds}:")
    print(f"{'-'*100}")


def plot_coe_distribution_from_array(array_path=None, normalization=False, coe_type=None, block_sz=2, reorder=False):
    coe_array = np.load(array_path)  # input array is 8*8 coe block
    print(f"{coe_array.shape} loaded")

    if normalization:
        print(f"use normalization")
        assert coe_type.lower() in ['rgb', 'y', 'cb', 'cr']

        if coe_type.lower() == 'rgb':
            coe_array = coe_array / 255.0
            coe_array = (coe_array - 0.5) / 0.5

        elif coe_type.lower() == 'y':
            coe_array = coe_array / CELEBA64_Y_bound
        elif coe_type.lower() == 'cb':
            coe_array = coe_array / CELEBA64_Cb_bound
        elif coe_type.lower() == 'cr':
            coe_array = coe_array / CELEBA64_Cr_bound
        else:
            raise ValueError(f"{coe_type} not implemented")


    stds = []
    entropys = []
    fig, axs = plt.subplots(1, 4, figsize=(20, 3))  # Create 1 row and 4 columns
    for i in range(block_sz**2):
        DCT_coe = coe_array[:, i]
        std = np.around(np.std(DCT_coe), decimals=3)
        stds.append(std)

        # percentile
        low_thresh = 1.5
        up_thresh = 100 - low_thresh
        lower_bound = np.percentile(DCT_coe, low_thresh)
        upper_bound = np.percentile(DCT_coe, up_thresh)
        filtered_coe = DCT_coe[(DCT_coe > lower_bound) & (DCT_coe < upper_bound)]

        # approximate the entropy by histogram
        counts, bin_edges = np.histogram(filtered_coe, bins=100, )
        probabilities = counts / np.sum(counts)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small epsilon to avoid log(0)
        entropys.append(entropy)
        print(f"{i} coe has entropy {entropy}")

        # Plotting the distribution
        axs[i].hist(filtered_coe, bins=50, color='ForestGreen', range=(-1, 1))
        # axs[i].set_xlabel('DCT coe')
        axs[i].set_ylabel('count')

        if i == 0:
            axs[i].set_title(f'Histogram of $D(0, 0)$')
        elif i == 1:
            axs[i].set_title(f'Histogram of $D(0, 1)$')
        elif i == 2:
            axs[i].set_title(f'Histogram of $D(1, 0)$')
        elif i == 3:
            axs[i].set_title(f'Histogram of $D(1, 1)$')
        axs[i].grid(True)

    fig.tight_layout()
    plt.savefig(f"/home/mang/coe_distribution.png")
    plt.show()
    plt.close()

    # stds_arr = np.array(stds).reshape(8, 8)
    # np.set_printoptions(suppress=True)
    # stds_arr = np.around(stds_arr, decimals=3)
    # print(f"std is {stds_arr}")
    entropys = np.array(entropys)
    np.set_printoptions(suppress=True)
    entropys = np.around(entropys, decimals=3)
    print(f"entropy is {list(entropys)}")


if __name__ == "__main__":
    # celeba_to_celeba64x64(input_folder='/home/mang/Downloads/celeba/img_align_celeba',
    #                       output_folder='/home/mang/Downloads/celeba/celeba64')
    # DCT_and_InverDCT(img_path='/home/mang/Downloads/celeba/celeba64/000001.jpg',
    #                  de_DCT=True)
    # images_to_array(image_folder='/home/mang/Downloads/celeba/celeba64',
    #                 save_path='/home/mang/Downloads/celeba64_normalized.npy')
    # image_to_DCT_array(img_folder='/home/mang/Downloads/celeba/celeba64',
    #                        block_sz=2)
    # DCT_statis_from_array(array_path='/home/mang/Downloads/celeba64_2by2_cr.npy', block_sz=2)
    plot_coe_distribution_from_array(array_path='/home/mang/Downloads/celeba64_2by2_y.npy',
                                     normalization=True, block_sz=2, coe_type='y')
