import os
import pickle
import numpy as np
from PIL import Image


def pickle_to_jpg_tra(data_folder=None, out_folder=None, img_sz=64):
    os.makedirs(out_folder, exist_ok=True)
    img_cnt = 1

    for i in range(1, 11):
        data_path = os.path.join(data_folder, f'train_data_batch_{i}')

        with open(data_path, 'rb') as fo:
            dict = pickle.load(fo)
            x = dict['data']
            labels = dict['labels']  # list
            assert x.shape[0] == len(labels)
            print(f"found {x.shape[0]} images with shape {x.shape} in folder {data_path}")

            # reshape data into 4D array (num_images, 64, 64, 3)
            channel_pixel = img_sz * img_sz
            x = np.dstack((x[:, :channel_pixel], x[:, channel_pixel:2*channel_pixel], x[:, 2*channel_pixel:]))
            x = x.reshape((x.shape[0], img_sz, img_sz, 3))

            # save images
            for j in range(x.shape[0]):
                img = Image.fromarray(x[j])
                img.save(os.path.join(out_folder, f'{labels[j]}_img_{img_cnt}.jpg'))
                img_cnt += 1
            print(f"Saved {x.shape[0]} images to {out_folder}")

    print(f"{len(os.listdir(out_folder))} images found in {out_folder}")


def pickle_to_jpg_val(data_folder=None, out_folder=None, img_sz=64):
    os.makedirs(out_folder, exist_ok=True)
    img_cnt = 1
    data_path = os.path.join(data_folder, 'val_data')

    with open(data_path, 'rb') as fo:
        dict = pickle.load(fo)
        x = dict['data']
        labels = dict['labels']  # list
        assert x.shape[0] == len(labels)
        print(f"found {x.shape[0]} images with shape {x.shape} in folder {data_path}")

        # reshape data into 4D array (num_images, 64, 64, 3)
        channel_pixel = img_sz * img_sz
        x = np.dstack((x[:, :channel_pixel], x[:, channel_pixel:2*channel_pixel], x[:, 2*channel_pixel:]))
        x = x.reshape((x.shape[0], img_sz, img_sz, 3))

        # save images
        for j in range(x.shape[0]):
            img = Image.fromarray(x[j])
            img.save(os.path.join(out_folder, f'{labels[j]}_img_{img_cnt}.jpg'))
            img_cnt += 1
        print(f"Saved {x.shape[0]} images to {out_folder}")


if __name__ == "__main__":
    pickle_to_jpg_tra(data_folder='/home/mang/Downloads', out_folder='/home/mang/Downloads/train')
    pickle_to_jpg_val(data_folder='/home/mang/Downloads', out_folder='/home/mang/Downloads/val')
