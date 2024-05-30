import numpy as np
import cv2


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


# if __name__ == "__main__":
#     print(zigzag_order(block_sz=8))
#     print(reverse_zigzag_order(block_sz=8))