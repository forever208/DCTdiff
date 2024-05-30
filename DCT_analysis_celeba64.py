from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


#########################
##### block size = 8 ####
#########################
# CELEBA64_Y_mean = np.array(
#     [-103.328, 0.636, 3.386, 0.127, 0.632, 0.016, 0.138, 0.006, -4.249, -0.054, 0.434, 0.028, -0.145, 0.003, -0.013, 0.001, 0.036, 0.049, -1.293, -0.032, -0.698, 0.007, -0.005, -0.003, 1.195, 0.039, -0.063, -0.005, 0.145, 0.001, 0.01, -0.001, -0.957, -0.003, -0.251, 0.021, 0.332, 0.0, 0.001, 0.001, -0.726, -0.011, -0.019, -0.012, -0.047, 0.001, -0.001, 0.0, -0.238, -0.003, 0.061, 0.003, -0.012, -0.0, -0.0, 0.0, -0.11, 0.003, -0.008, 0.006, 0.007, -0.0, 0.0, 0.0]
# )
#
# CELEBA64_Cb_mean = np.array(
#     [-106.469, 0.002, -1.302, 0.016, -0.028, 0.001, 0.006, 0.001, 1.572, 0.007, 0.704, -0.001, 0.002, 0.0, -0.001, -0.0, 0.87, -0.026, -0.074, 0.004, -0.002, 0.01, 0.004, 0.004, 0.779, -0.001, -0.095, 0.001, 0.001, 0.001, -0.0, 0.001, -0.045, 0.004, 0.004, 0.007, 0.006, 0.003, 0.002, 0.003, 0.042, -0.0, -0.047, 0.001, -0.001, 0.0, 0.001, 0.0, 0.027, 0.003, -0.009, 0.004, 0.001, 0.002, 0.0, 0.001, 0.001, 0.001, -0.01, -0.0, -0.0, -0.0, 0.001, 0.0]
# )
#
# CELEBA64_Cr_mean = np.array(
#     [137.019, -0.204, 2.476, -0.151, 0.003, -0.019, -0.004, -0.003, -1.974, 0.007, -0.571, 0.004, -0.003, -0.001, 0.0, -0.001, -0.733, 0.015, 0.019, -0.004, -0.002, 0.005, -0.001, 0.001, -1.371, 0.002, 0.062, -0.001, -0.001, 0.0, -0.0, -0.0, 0.082, 0.002, -0.005, 0.007, -0.001, 0.004, -0.002, 0.002, -0.096, -0.0, 0.033, -0.0, 0.002, 0.0, -0.001, 0.0, -0.006, 0.003, 0.005, 0.004, 0.0, 0.002, -0.002, 0.001, -0.029, -0.0, 0.011, -0.0, -0.0, 0.0, -0.0, -0.0]
# )
#
#
# # bound of 99% interval
# CELEBA64_Y_99_centered = np.array(
#     [1053.544, 503.066, 256.474, 144.344, 96.41, 60.408, 51.853, 31.429, 410.352, 204.285, 147.134, 99.762, 66.39, 57.843, 30.595, 28.398, 251.964, 126.276, 101.247, 72.38, 60.528, 30.831, 35.356, 28.3, 133.578, 89.345, 66.151, 59.305, 51.746, 44.143, 39.971, 30.985, 90.728, 65.729, 56.828, 31.382, 34.697, 3.899, 2.063, 2.196, 60.389, 52.898, 28.743, 32.373, 40.856, 1.81, 1.459, 1.405, 49.494, 32.4, 39.1, 42.387, 1.683, 1.36, 1.299, 1.255, 36.14, 5.652, 2.054, 1.687, 1.396, 1.286, 1.235, 1.195]
# )
#
# CELEBA64_Cb_99_centered = np.array(
#     [276.705, 112.847, 53.747, 23.107, 4.106, 2.639, 2.19, 1.836, 85.239, 42.732, 33.256, 26.639, 2.587, 1.989, 1.669, 1.407, 35.583, 24.347, 23.989, 2.729, 2.046, 1.634, 1.475, 1.296, 21.215, 23.632, 2.529, 2.038, 1.61, 1.398, 1.302, 1.211, 2.599, 1.929, 1.843, 1.565, 1.342, 1.172, 1.164, 1.078, 1.756, 1.514, 1.352, 1.246, 1.083, 1.017, 1.001, 0.95, 1.556, 1.257, 1.248, 1.138, 1.053, 0.982, 0.997, 0.955, 1.548, 1.063, 1.126, 1.027, 0.959, 0.915, 0.928, 1.086]
# )
#
# CELEBA64_Cr_99_centered = np.array(
#     [276.277, 123.739, 55.896, 25.801, 4.289, 2.49, 1.911, 1.547, 95.044, 51.402, 35.06, 27.007, 2.339, 1.777, 1.355, 1.032, 44.671, 25.865, 24.259, 2.645, 1.899, 1.414, 1.158, 0.944, 22.097, 26.114, 2.393, 2.027, 1.462, 1.196, 1.012, 0.893, 3.295, 2.232, 1.876, 1.528, 1.219, 0.966, 0.908, 0.818, 2.011, 1.746, 1.431, 1.238, 0.998, 0.863, 0.81, 0.754, 1.619, 1.38, 1.254, 1.046, 0.909, 0.815, 0.822, 0.755, 1.405, 1.056, 0.978, 0.892, 0.818, 0.745, 0.764, 0.885]
# )
#
#
# Y_std = np.array(
#     [450.37, 149.152, 69.349, 38.811, 24.917, 17.109, 11.477, 7.624, 115.158, 62.444, 40.163, 27.189, 19.39, 14.023, 9.692, 6.769, 59.845, 37.219, 27.721, 20.96, 16.643, 12.182, 8.366, 6.09, 33.018, 24.043, 18.586, 15.442, 12.303, 8.497, 6.002, 4.661, 22.2, 16.933, 14.469, 11.74, 9.302, 5.133, 3.447, 3.057, 15.126, 12.271, 10.178, 8.141, 5.754, 3.333, 1.86, 1.694, 9.962, 8.278, 6.541, 4.593, 2.837, 1.518, 0.987, 0.891, 5.965, 4.552, 3.405, 2.798, 1.654, 1.291, 0.82, 0.666]
# )
#
# Cb_std = np.array(
#     [81.27, 32.73, 15.34, 8.145, 2.629, 1.106, 0.667, 0.521, 22.818, 13.912, 9.416, 4.333, 1.311, 0.686, 0.5, 0.402, 10.711, 7.892, 4.707, 1.519, 0.779, 0.514, 0.424, 0.369, 5.579, 2.763, 1.239, 0.746, 0.49, 0.398, 0.368, 0.34, 1.711, 0.896, 0.671, 0.494, 0.382, 0.339, 0.322, 0.311, 0.701, 0.525, 0.448, 0.37, 0.318, 0.3, 0.287, 0.276, 0.476, 0.395, 0.376, 0.34, 0.3, 0.286, 0.287, 0.276, 0.393, 0.332, 0.322, 0.301, 0.285, 0.268, 0.272, 0.286]
# )
#
# Cr_std = np.array(
#     [94.452, 39.289, 16.906, 9.503, 3.331, 1.325, 0.69, 0.475, 27.569, 15.69, 10.561, 5.316, 1.671, 0.78, 0.476, 0.352, 12.676, 9.222, 5.656, 1.895, 0.94, 0.523, 0.388, 0.317, 6.887, 4.107, 1.628, 0.925, 0.538, 0.378, 0.324, 0.287, 2.372, 1.355, 0.815, 0.546, 0.374, 0.304, 0.281, 0.266, 0.907, 0.685, 0.482, 0.386, 0.302, 0.273, 0.257, 0.244, 0.518, 0.441, 0.38, 0.329, 0.279, 0.257, 0.254, 0.242, 0.378, 0.333, 0.302, 0.277, 0.257, 0.239, 0.241, 0.247]
# )



#########################
##### block size = 4 ####
#########################

# # before reorder
# CELEBA64_Y_mean = np.array(
#     [-51.986, 0.181, 0.316, 0.008, -0.298, 0.008, 0.037, -0.001, -0.479, 0.008, 0.166, -0.0, -0.333, -0.002, -0.017, 0.0]
# )
#
# CELEBA64_Cb_mean = np.array(
#     [-52.955, 0.007, -0.014, -0.0, 0.628, 0.0, 0.001, 0.0, -0.022, 0.003, 0.003, 0.001, -0.009, 0.0, -0.0, 0.0]
# )
#
# CELEBA64_Cr_mean = np.array(
#     [68.254, -0.099, 0.002, -0.001, -0.94, 0.001, -0.002, -0.0, 0.041, 0.003, -0.001, 0.002, 0.008, -0.0, 0.001, -0.0]
# )
#
#
# # bound of 99% interval
# CELEBA64_Y_99_centered = np.array(
#     [541.102, 208.157, 90.854, 43.774, 176.104, 82.552, 52.65, 31.538, 87.123, 48.111, 32.788, 17.292, 41.344, 25.836, 13.897, 3.857]
# )
#
# CELEBA64_Cb_99_centered = np.array(
#     [156.585, 42.231, 11.461, 2.799, 32.131, 15.096, 3.573, 1.499, 8.67, 2.834, 1.518, 1.164, 2.186, 1.259, 1.058, 1.001]
# )
#
# CELEBA64_Cr_99_centered = np.array(
#     [154.203, 47.668, 13.927, 2.895, 37.978, 16.792, 3.94, 1.275, 11.573, 3.409, 1.486, 0.903, 2.528, 1.259, 0.919, 0.818]
# )
#
#
# # std before normalization
# Y_std = np.array(
#     [226.619, 52.408, 21.509, 10.019, 44.651, 21.412, 13.092, 6.965, 19.061, 11.785, 7.574, 3.53, 8.61, 5.474, 2.977, 1.236]
# )
#
# Cb_std = np.array(
#     [44.396, 11.885, 3.072, 0.823, 8.55, 3.67, 0.966, 0.413, 2.195, 0.838, 0.411, 0.314, 0.619, 0.375, 0.297, 0.284]
# )
#
# Cr_std = np.array(
#     [51.96, 13.756, 3.623, 0.911, 10.214, 4.329, 1.15, 0.4, 2.792, 1.023, 0.439, 0.28, 0.726, 0.392, 0.278, 0.251]
# )



#########################
##### block size = 2 ####
#########################

# # before reorder
# CELEBA64_Y_mean = np.array(
#     [-20.702, 0.037, -0.211, 0.0]
# )
#
# CELEBA64_Cb_mean = np.array(
#     [-21.787, 0.001, 0.116, 0.0]
# )
#
# CELEBA64_Cr_mean = np.array(
#     [27.142, -0.019, -0.176, 0.0]
# )
#
#
# # bound of 99% interval
# CELEBA64_Y_99_centered = np.array(
#     [266.999, 72.75, 63.072, 23.263]
# )
#
# CELEBA64_Cb_99_centered = np.array(
#     [77.388, 10.552, 8.196, 1.812]
# )
#
# CELEBA64_Cr_99_centered = np.array(
#     [82.391, 11.963, 9.729, 1.946]
# )
#
#
# # std before normalization
# Y_std = np.array(
#     [77.953, 15.432, 13.612, 5.06]
# )
#
# Cb_std = np.array(
#     [21.924, 2.8, 2.086, 0.468]
# )
#
# Cr_std = np.array(
#     [26.072, 3.28, 2.514, 0.516]
# )



# before reorder
CELEBA64_Y_mean = np.array(
    [-20.702, 0.037, -0.211, 0.0]
)

CELEBA64_Cb_mean = np.array(
    [-21.787, 0.001, 0.116, 0.0]
)

CELEBA64_Cr_mean = np.array(
    [27.142, -0.019, -0.176, 0.0]
)


# bound of 99% interval without mean subtraction
CELEBA64_Y_99_centered = np.array(
    [250.298, 250.298, 250.298, 250.298]
)

CELEBA64_Cb_99_centered = np.array(
    [250.298, 250.298, 250.298, 250.298]
)

CELEBA64_Cr_99_centered = np.array(
    [250.298, 250.298, 250.298, 250.298]
)


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

    """statistics for 99percentile, mean and std"""
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
        # lower_bound = np.percentile(coe_array[:, index] - means[index], 0.5)
        # upper_bound = np.percentile(coe_array[:, index] - means[index], 99.5)
        lower_bound = np.percentile(coe_array[:, index] - means[index], 1.25)
        upper_bound = np.percentile(coe_array[:, index] - means[index], 98.75)
        lower_bound = np.round(lower_bound, 3)
        upper_bound = np.round(upper_bound, 3)
        print(f"(99%) coe {index} has upper bound {upper_bound} and lower bound {lower_bound} after subtracting mean {means[index]}")

        if np.abs(lower_bound) > np.abs(upper_bound):
            DCT_coe_bounds.append(np.abs(lower_bound))
        else:
            DCT_coe_bounds.append(np.abs(upper_bound))
    print(f"99percentile bound is {DCT_coe_bounds}:")
    print(f"{'-'*100}")


def plot_coe_distribution_from_array(array_path=None, normalization=False, coe_type=None, block_sz=2, reorder=False):
    coe_array = np.load(array_path)  # input array is 8*8 coe block
    print(f"{coe_array.shape} loaded")

    # take log with log base 100
    # coe_array[coe_array > 0] = np.log10(coe_array[coe_array > 0] + 1) / np.log10(100)
    # coe_array[coe_array < 0] = -(np.log10(-coe_array[coe_array < 0] + 1) / np.log10(100))

    if normalization:
        print(f"use normalization")
        assert coe_type.lower() in ['rgb', 'y', 'cb', 'cr']

        if coe_type.lower() == 'rgb':
            coe_array = coe_array / 255.0
            coe_array = (coe_array - 0.5) / 0.5

        elif coe_type.lower() == 'y':
            coe_array = (coe_array - CELEBA64_Y_mean) / CELEBA64_Y_99_centered
        elif coe_type.lower() == 'cb':
            coe_array = (coe_array - CELEBA64_Cb_mean) / CELEBA64_Cb_99_centered
        elif coe_type.lower() == 'cr':
            coe_array = (coe_array - CELEBA64_Cr_mean) / CELEBA64_Cr_99_centered
        else:
            raise ValueError(f"{coe_type} not implemented")


    stds = []
    entropys = []
    for i in range(block_sz**2):
        DCT_coe = coe_array[:, i]
        std = np.around(np.std(DCT_coe), decimals=3)
        stds.append(std)

        # 99%
        lower_bound = np.percentile(DCT_coe, 0.5)
        upper_bound = np.percentile(DCT_coe, 99.5)
        filtered_coe = DCT_coe[(DCT_coe > lower_bound) & (DCT_coe < upper_bound)]

        # compute the entroy
        # from scipy.stats import gaussian_kde
        # from scipy.integrate import quad
        #
        # # KDE to estimate the PDF
        # kde = gaussian_kde(filtered_coe)
        #
        # def integrand(x):
        #     pdf_value = kde.evaluate(x)[0]
        #     if pdf_value > 0:  # To avoid log(0)
        #         return -pdf_value * np.log(pdf_value)
        #     else:
        #         return 0
        #
        # # Compute the integral
        # entropy, error = quad(integrand, -1, 1)
        # entropys.append(entropy)
        # print(f"{i} coe has entropy {entropy}")

        # approximate the entropy by histogram
        counts, bin_edges = np.histogram(filtered_coe, bins=100, )
        probabilities = counts / np.sum(counts)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small epsilon to avoid log(0)
        entropys.append(entropy)
        print(f"{i} coe has entropy {entropy}")

        # Plotting the distribution
        plt.figure(figsize=(10, 6))
        plt.hist(filtered_coe, bins=50, alpha=0.75, color='green', edgecolor='black')
        plt.xlabel('DCT coe')
        plt.ylabel('Frequency')
        plt.title(f'DCT coe Distribution at {i}')
        plt.grid(True)
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(f"/home/mang/{i}.png")
        plt.close()
        # plt.show()

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
    #                        block_sz=8)
    # image_to_DCT_array(img_folder='/home/mang/Downloads/celeba/celeba64',
    #                        block_sz=2)
    DCT_statis_from_array(array_path='/home/mang/Downloads/celeba64_2by2_y.npy', block_sz=2)
    # plot_coe_distribution_from_array(array_path='/home/mang/Downloads/celeba64_2by2_cb.npy',
    #                                  normalization=True, block_sz=2, coe_type='cb')
