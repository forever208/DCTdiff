import numpy as np
import cv2
import torch
import torch_dct as dct


# def split_into_blocks(image, block_sz):
#     blocks = []
#     for i in range(0, image.shape[0], block_sz):
#         for j in range(0, image.shape[1], block_sz):
#             blocks.append(image[i:i + block_sz, j:j + block_sz])  # first row, then column
#     return np.array(blocks)
#
#
# def combine_blocks(blocks, height, width, block_sz):
#     image = np.zeros((height, width), np.float32)
#     index = 0
#     for i in range(0, height, block_sz):
#         for j in range(0, width, block_sz):
#             image[i:i + block_sz, j:j + block_sz] = blocks[index]
#             index += 1
#     return image
#
#
# def dct_transform(blocks):
#     dct_blocks = []
#     for block in blocks:
#         dct_block = np.float32(block) - 128  # Shift to center around 0
#         dct_block = cv2.dct(dct_block)
#         dct_blocks.append(dct_block)
#     return np.array(dct_blocks)
#
#
# def idct_transform(blocks):
#     idct_blocks = []
#     for block in blocks:
#         idct_block = cv2.idct(block)
#         idct_block = idct_block + 128  # Shift back
#         idct_blocks.append(idct_block)
#     return np.array(idct_blocks)


def zigzag_order(block_sz=8):
    index_list = []

    # Iterate over each diagonal defined by the sum of row and column indices
    for s in range(2 * (block_sz - 1) + 1):
        temp = []  # Initialize a temporary list to collect indices in the current diagonal
        start = max(0, s - block_sz + 1)  # Calculate starting and ending points of the diagonal
        end = min(s, block_sz - 1)

        for i in range(start, end + 1):  # Collect indices in the current diagonal
            temp.append((i, s - i))

        if s % 2 == 0:  # Reverse the diagonal elements if the sum of indices is even
            temp.reverse()

        index_list.extend(temp)  # Convert 2D indices to 1D and append to the main list

    return [i * block_sz + j for i, j in index_list]  # Convert tuple (i, j) to index i * B + j


def reverse_zigzag_order(block_sz=8):
    zigzag_indices = zigzag_order(block_sz)  # Get the zigzag order list
    reverse_order = [0] * (block_sz * block_sz)  # Initialize an array of the same size to store the reverse order

    # Populate the reverse order list where the index is the original position,
    # and the value is the new position according to the zigzag order
    for index, value in enumerate(zigzag_indices):
        reverse_order[value] = index

    return reverse_order


def split_into_blocks_torch(image: torch.Tensor, block_sz: int):
    """
    Split a 2D tensor (H, W) or batched 3D tensor (B, H, W) into non-overlapping (block_sz x block_sz) blocks.

    Args:
        image (Tensor): shape (H, W) or (B, H, W)
        block_sz (int): block size

    Returns:
        Tensor:
            - (N_blocks, block_sz, block_sz) if input is (H, W)
            - (B, N_blocks, block_sz, block_sz) if input is (B, H, W)
    """
    if image.dim() == 2:  # Single image
        H, W = image.shape
        assert H % block_sz == 0 and W % block_sz == 0
        blocks = image.unfold(0, block_sz, block_sz).unfold(1, block_sz, block_sz)  # (H/B, W/B, B, B)
        return blocks.contiguous().view(-1, block_sz, block_sz)  # (N_blocks, B, B)

    elif image.dim() == 3:  # Batched images
        B, H, W = image.shape
        assert H % block_sz == 0 and W % block_sz == 0
        blocks = image.unfold(1, block_sz, block_sz).unfold(2, block_sz, block_sz)  # (B, H/B, W/B, B, B)
        blocks = blocks.contiguous().view(B, -1, block_sz, block_sz)  # (B, N_blocks, B, B)
        return blocks

    else:
        raise ValueError(f"Input tensor must be 2D or 3D, got shape {image.shape}")


def combine_blocks_torch(blocks: torch.Tensor, height: int, width: int, block_sz: int):
    """
    Combine non-overlapping blocks into full image.

    Args:
        blocks:
            - (N, B, B) tensor (single image)
            - (batch, N, B, B) tensor (batched images)
        height: original image height
        width: original image width
        block_sz: size of each block (B)

    Returns:
        image:
            - (height, width) if input is 3D
            - (batch, height, width) if input is 4D
    """
    blocks_per_row = width // block_sz
    blocks_per_col = height // block_sz

    if blocks.dim() == 3:  # (N, B, B)
        image = blocks.view(blocks_per_col, blocks_per_row, block_sz, block_sz)
        image = image.permute(0, 2, 1, 3).reshape(height, width)
        return image

    elif blocks.dim() == 4:  # (batch, N, B, B)
        B = blocks.size(0)
        image = blocks.view(B, blocks_per_col, blocks_per_row, block_sz, block_sz)
        image = image.permute(0, 1, 3, 2, 4).reshape(B, height, width)
        return image

    else:
        raise ValueError(f"Expected input of shape (N, B, B) or (batch, N, B, B), but got {blocks.shape}")


def dct_2d_torch(x):
    # x: (B, H, W) or (H, W) float32 tensor

    # Apply 2D DCT Type-II
    x = x.float() - 128.0  # Ensure float32 and subtract 128 (OpenCV style)
    x = dct.dct(x, norm='ortho')                 # DCT along last dimension
    x = dct.dct(x.transpose(-2, -1), norm='ortho')  # DCT along second-last
    return x.transpose(-2, -1)


def idct_2d_torch(x):
    # x: (B, H, W) or (H, W) tensor, result of OpenCV-style DCT

    # Apply 2D IDCT (Type-III)
    x = dct.idct(x, norm='ortho')                    # inverse DCT on last dim
    x = dct.idct(x.transpose(-2, -1), norm='ortho')  # inverse DCT on second-last dim
    x = x.transpose(-2, -1)

    return x + 128.0  # Add 128 back to undo JPEG-style preprocessing


def Batch_DCT_to_RGB(DCT_tensor, block_sz=8, img_sz=128, eta=1):
    DCT_tensor = DCT_tensor * eta

    y_blocks = split_into_blocks_torch(DCT_tensor[:, 0, :, :], block_sz)  # (batch, h, w) --> (batch, h/B * w/B, B, B)
    cb_blocks = split_into_blocks_torch(DCT_tensor[:, 1, :, :], block_sz)
    cr_blocks = split_into_blocks_torch(DCT_tensor[:, 2, :, :], block_sz)

    y_blocks = idct_2d_torch(y_blocks)  # (batch, h/B * w/B, B, B)
    cb_blocks = idct_2d_torch(cb_blocks)
    cr_blocks = idct_2d_torch(cr_blocks)

    y_blocks = combine_blocks_torch(y_blocks, img_sz, img_sz, block_sz)  # (batch, h, w)
    cb_blocks = combine_blocks_torch(cb_blocks, img_sz, img_sz, block_sz)
    cr_blocks = combine_blocks_torch(cr_blocks, img_sz, img_sz, block_sz)

    rgb_recon = torch.stack([
        y_blocks + 1.402 * (cr_blocks - 128),  # R channel
        y_blocks - 0.344136 * (cb_blocks - 128) - 0.714136 * (cr_blocks - 128),  # G channel
        y_blocks + 1.772 * (cb_blocks - 128)  # B channel
    ], dim=1)  # (batch, 3, h, w), value range [0, 255]

    rgb_recon = rgb_recon / 255.0  # (batch, 3, h, w), value range [0, 1]

    return rgb_recon


if __name__ == "__main__":
    print(zigzag_order(block_sz=8))
    print(reverse_zigzag_order(block_sz=8))