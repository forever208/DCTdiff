from PIL import Image
import os


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


if __name__ == "__main__":
    celeba_to_celeba64x64(input_folder='/home/mang/Downloads/celeba/img_align_celeba',
                          output_folder='/home/mang/Downloads/celeba/celeba64')
