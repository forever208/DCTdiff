## DCTdiff

Official PyTorch implementation of our ICML 2025 paper: [DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](https://arxiv.org/abs/2412.15032)

The repository is based on [UViT](https://github.com/baofff/U-ViT). We keep the UViT code unchanged in the UViT branch where you can find our baseline implementation.

We will also release the DCTdiff based on [DiT](https://github.com/facebookresearch/DiT) in the near future


![img.png](img.png)



## Installation

The installation works both for DCTdiff and UViT


```sh
conda create -n dctdiff python==3.9
conda activate dctdiff

pip install matplotlib
pip install accelerate==0.33.0   # (auto install pytorch 2.4)
pip install absl-py ml_collections einops wandb ftfy==6.1.1 transformers==4.23.1
pip install opencv-python
pip install scipy

# xformers is optional, but it would greatly speed up the attention computation.
pip install -U xformers
pip install torchvision==0.19.0
```


## Trained Models (DCTdiff)
CIFAR-10: [cifar10_small.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_cifar10_small.pth), [cifar10_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_cifar10_mid.pth), [cifar10_mid_deep.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_cifar10_mid_deep.pth)

CelebA 64x64: [celeba64_small_DPMSolver.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_celeba64_small_DPMSolver.pth), [celeba64_small_ddim.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_celeba64_small_ddim.pth)

ImageNet 64x64: [imgnet64_small.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_imgnet64_small.pth), [imgnet64_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_imgnet64_mid.pth), [imgnet64_mid_deep.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_imgnet64_mid_deep.pth)

FFHQ 128x128: [ffhq128_small.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_ffhq128_small.pth), [ffhq128_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_ffhq128_mid.pth), [ffhq128_mid_deep.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_ffhq128_mid_deep.pth)

FFHQ 256x256: [ffhq256_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_ffhq256_mid.pth)

FFHQ 512x512: [ffhq512_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_ffhq512_mid.pth)

AFHQ 512x512: [afhq512_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/DCTdiff_afhq512_mid.pth)


## Trained Models (UViT)
CIFAR-10: [cifar10_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/UViT_cifar10_mid.pth)

CelebA 64x64: [celeba64_small.pth](https://huggingface.co/forever208/DCTdiff/blob/main/UViT_celeba64_small.pth)

ImageNet 64x64: [imgnet64_small.pth](https://huggingface.co/forever208/DCTdiff/blob/main/UViT_imgnet64_small.pth)

FFHQ 128x128: [ffhq128_small.pth](https://huggingface.co/forever208/DCTdiff/blob/main/UViT_ffhq128_small.pth)

FFHQ 256x256: [ffhq256_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/UViT_ffhq256_mid.pth)

FFHQ 512x512: [ffhq512_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/UViT_ffhq512_mid.pth)

AFHQ 512x512: [afhq512_mid.pth](https://huggingface.co/forever208/DCTdiff/blob/main/UViT_afhq512_mid.pth)



## Preparation Before Training and Evaluation

#### Data
Each dataset is organized in a 'folder' format. You can either use jpg or png for the datasets.

But note that, if you use png images for training, your generated images must be saved into png as well before computing FID.
In our experiments, we use jpg images and the fid_stats is computed from the jpg image folder.


* CIFAR-10 32x32: we provide a script (`tools/download_cifar10.py`) to download CIFAR-10
* CelebA 64x64: download the [dataset](https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&authuser=0), then do center crop to 64x64 using the script `tools/dataset_celeba64.py`.
* ImageNet 64x64: download the [dataset](https://www.kaggle.com/datasets/ayaroshevskiy/downsampled-imagenet-64x64), use the `train` folder for training.
* FFHQ 128x128: download the [dataset](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set)
* FFHQ 256x256: download the [dataset](https://www.kaggle.com/datasets/rahulbhalley/ffhq-256x256)
* FFHQ 512x512: download the [dataset](https://www.kaggle.com/datasets/shivamshrirao/ffhqresized512x512)
* AFHQ 512x512: download the [dataset](https://www.kaggle.com/datasets/dimensi0n/afhq-512?resource=download)


#### Reference statistics for FID
Download `fid_stats` directory from this [link](https://drive.google.com/drive/folders/1WvLoI73O1HXyleg2jWzginlFJ5eJ_s-b?usp=sharing) (which contains reference statistics for FID).

Put the downloaded `fid_stats` into directory `assets/fid_stats`, the path of `fid_stats` is set in the script `datasets.py` for FID comutation in both training and inference.

Using [pytorch-FID](https://github.com/mseitzer/pytorch-fid), you can also 
* generate your own fid_stats for a given dataset
* compute the FID whenever you need.

```shell
python -m pytorch_fid --save-stats path/to/dataset_folder path/to/fid_stats  # generate fid_stats
python -m pytorch_fid path/to/dataset1_folder/to/dataset2_folder  # FID calculation
```


#### Parameters (eta, entropy, m*)
We have provided all DCT-related parameters in the config files `configs`,  

If you need to train on the other datasets, please use the script `DCT_datasets_statis.py` to compute `eta` and `entropy`, or determine `m*`


## Training

We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library to help train with distributed data parallel and mixed precision.

We provide all commands to reproduce DCTdiff training in the paper (4xA100 are used for all experiments):

Feel free to change the sampler, NFE and num_samples in the config file.

```sh
# CIFAR10 32x32
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/cifar10_uvit_mid_2by2.py --workdir YOUR_DIR

# CelebA 64x64
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/celeba64_uvit_small_2by2.py --workdir YOUR_DIR

# ImageNet 64x64
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/imgnet64_uvit_small_2by2.py --workdir YOUR_DIR

# FFHQ 128x128
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/ffhq128_uvit_small_4by4.py --workdir YOUR_DIR

# FFHQ 256x256
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/ffhq256_uvit_mid_4by4.py --workdir YOUR_DIR

# FFHQ 512x512
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/ffhq512_uvit_mid_8by8.py --workdir YOUR_DIR

# AFHQ 512x512
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/afhq512_uvit_mid_8by8.py --workdir YOUR_DIR
```



## FID Evaluation

We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library for efficient inference with mixed precision and multiple gpus. The following is the evaluation command:

An example of evaluating the FID on CIFAR-10 is shown below. Evaluation on other datasets just require to change the --config and --nnet_path accordingly,
```sh
# CIFAR10 32x32
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/cifar10_uvit_mid_2by2.py --nnet_path=DCTdiff_cifar10_mid.pth --output_path YOUR_DIR
```




## References
If you find the code useful for your research, please consider citing
```bib
@article{ning2024dctdiff,
  title={DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space},
  author={Ning, Mang and Li, Mingxiao and Su, Jianlin and Jia, Haozhe and Liu, Lanmiao and Bene{\v{s}}, Martin and Salah, Albert Ali and Ertugrul, Itir Onal},
  journal={arXiv preprint arXiv:2412.15032},
  year={2024}
}
```