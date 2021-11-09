from math import log10, sqrt
from typing import Sequence
import numpy as np
from . import train
from vanilla_vae.features import utils
import torch
import os
import sys
import yaml
from torchvision.transforms import ToTensor, ToPILImage
import pathlib
import PIL
import matplotlib.pyplot as plt
import argparse


def PSNR(original, restored):
    mse = np.mean((original - restored) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = np.max(restored)
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def psnr_plot(path_to_folder, image_sample, height, width, gt):
    psnr_list = []
    models_list_length = len(sorted(path_to_folder.rglob("*.net")))
    print(models_list_length)
    for i in range(models_list_length):
        model_name = "epoch-" + str(i) + ".net"
        net = torch.load(path_to_folder / model_name)
        output = utils.predictMMSE(image_sample, 10, net, size=(height, width))
        psnr_list.append(PSNR(gt, output))
    
    return np.array(psnr_list)

# folder to load config file
CONFIG_PATH = "/home/fahad/master_thesis/vanilla_vae/experiments/"

def eval(
    config_name: str,
    multi_gpu: bool,
    epoch: str,
    dataset_type: str,
    plots: bool,
    noise_type: str
):
    config_file = train.load_config(config_name + ".yaml")


    checkpoints_path = os.path.join(config_file["save_default_path"], config_file["title"]) + "/checkpoints/"
    plots_path = os.path.join(config_file["save_default_path"], config_file["title"]) + "/plots/"
    images_folder_path = pathlib.Path(config_file["data_directory"]) / "val"

    os.mkdir(plots_path + noise_type)

    image_path = utils.load_paths(
        data_directory=images_folder_path,
        n_samples=101,
    )[100]

    parallel_network = multi_gpu

    if noise_type=="gaussian":
        gaussian = True
    if noise_type=="dot_noise":
        synthetic = True
    else:
        print("Noise not recognized")
        return None

    dot_noise_index = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    gaussian_noise_index = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.09]

    image = utils.load_images([image_path])

    for dot_noise, gaussian_noise in zip(dot_noise_index, gaussian_noise_index):
        image_transform = utils.Noising(
            gaussian=gaussian,
            particle_noise=synthetic,
            seed=config_file["seed"],
            clip=config_file["clip"],
            gaussian_variance=gaussian_noise,
            particle_noise_density=dot_noise,
        )
        test_image = image_transform.noising(images_list=image)[0]

        height, width = test_image.shape
        image_sample = utils.convertNumpyToTensor(test_image).view(1,1,height,width).cuda()

        net = torch.load(checkpoints_path + "epoch-" + epoch + ".net")

        output = utils.predictMMSE(image_sample, 10, net, size=(height, width), parallel_network=parallel_network)
       
        display_output(
            test_image=test_image,
            output=output,
            ground_truth=image[0],
            config_file=config_file,
            noise_type=noise_type
        )
        
        if plots:
            psnr_data = psnr_plot(
                path_to_folder=pathlib.Path(checkpoints_path),
                image_sample=image_sample,
                height=height,
                width=width,
                gt=image,
            )

            np.save(plots_path + str(dot_noise) + "psnr_data.npy", psnr_data)


def display_output(test_image, output, ground_truth, config_file, image_name, noise_type):
    fig, axes =plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

    axes[0].set_title("Raw")
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_xticks(())
    axes[0].set_yticks(())

    axes[1].set_title("Output")
    axes[1].imshow(output, cmap='gray')
    axes[1].set_xticks(())
    axes[1].set_yticks(())

    axes[2].set_title("Ground Truth")
    axes[2].imshow(ground_truth, cmap='gray')
    axes[2].set_xticks(())
    axes[2].set_yticks(())

    fig.tight_layout()
    fig.savefig(os.path.join(config_file["save_default_path"], config_file["title"]) + "/plots/" + noise_type + "/" + image_name + ".png")
    




