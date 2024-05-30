import numpy as np
import os
from PIL import Image
import shutil
import random


def images_to_npz(img_folder=None, output_npz=None):
    # given the training image folder, generate the reference batch for evaluation
    images_list = []

    for filename in os.listdir(img_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
            file_path = os.path.join(img_folder, filename)

            with Image.open(file_path) as img:
                image_array = np.array(img)
                images_list.append(image_array)

    if images_list:
        all_images_array = np.stack(images_list, axis=0)
        np.savez(output_npz, all_images_array)
        print(f"All images have been saved to {output_npz} with shape {all_images_array.shape}")
    else:
        print("No images to save.")


def randomly_draw_imgs(source_folder=None, destination_folder=None, num_images=10000):
    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # List all image files in the source folder
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    all_images = [f for f in os.listdir(source_folder)
                  if os.path.isfile(os.path.join(source_folder, f)) and os.path.splitext(f)[1].lower() in valid_extensions]

    if len(all_images) < num_images:
        raise ValueError(f"Only {len(all_images)} images found, but {num_images} requested.")

    selected_images = random.sample(all_images, num_images)

    # Copy selected images to the destination folder
    for img in selected_images:
        src_path = os.path.join(source_folder, img)
        dst_path = os.path.join(destination_folder, img)
        shutil.copy2(src_path, dst_path)

    print(f"{num_images} images successfully copied to {destination_folder}")



def copy_all_images_to_a_folder(source_root=None, target_folder=None, rename_to_avoid_conflicts=True):
    os.makedirs(target_folder, exist_ok=True)

    for subfolder in os.listdir(source_root):
        subfolder_path = os.path.join(source_root, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                source_file = os.path.join(subfolder_path, filename)
                if os.path.isfile(source_file):
                    if rename_to_avoid_conflicts:
                        target_file = os.path.join(target_folder, f"{subfolder}_{filename}")
                    else:
                        target_file = os.path.join(target_folder, filename)
                    shutil.copy2(source_file, target_file)

    print(f"All images copied from '{source_root}' to '{target_folder}'.")


if __name__ == "__main__":
    # CIFAR-10
    images_to_npz(img_folder='/data/scratch/datasets/cifar10/cifar_train',
                  output_npz='/data/scratch/datasets/cifar_train.npz')
    randomly_draw_imgs(source_folder='/data/scratch/datasets/cifar10/cifar_train',
                       destination_folder='/data/scratch/datasets/cifar10_10k',
                       num_images=10000)

    # FFHQ 128
    images_to_npz(img_folder='/data/scratch/datasets/ffhq128/ffhq128',
                  output_npz='/data/scratch/datasets/ffhq128.npz')
    randomly_draw_imgs(source_folder='/data/scratch/datasets/ffhq128/ffhq128',
                       destination_folder='/data/scratch/datasets/ffhq128_10k',
                       num_images=10000)

    # AFHQ 512
    copy_all_images_to_a_folder(source_root='/data/scratch/datasets/afhq512_jpg',
                                target_folder='/data/scratch/datasets/afhq512_single_folder',
                                rename_to_avoid_conflicts=True)
    images_to_npz(img_folder='/data/scratch/datasets/afhq512_single_folder',
                  output_npz='/data/scratch/datasets/afhq512.npz')
    randomly_draw_imgs(source_folder='/data/scratch/datasets/afhq512_single_folder',
                       destination_folder='/data/scratch/datasets/afhq512_10k',
                       num_images=10000)