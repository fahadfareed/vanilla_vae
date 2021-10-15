from math import log10, sqrt
from typing import Sequence
import numpy as np
from src.features import utils
import torch
import os
import sys
import yaml
from torchvision.transforms import ToTensor, ToPILImage
import pathlib
import PIL
import matplotlib.pyplot as plt


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
CONFIG_PATH = "../../experiments/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config_file = yaml.safe_load(file)

    return config_file

config_file = load_config(str(sys.argv[1]) + ".yaml")
checkpoints_path = os.path.join(config_file["save_default_path"], config_file["title"]) + "/checkpoints/"
plots_path = os.path.join(config_file["save_default_path"], config_file["title"]) + "/plots/"
images_folder_path = pathlib.Path(config_file["data_directory"]) / "val"

image_path = utils.load_paths(
    data_directory=images_folder_path,
    n_samples=101,
)[100]

epoch = sys.argv[2]
parallel_network = sys.argv[3]
gaussian = sys.argv[4]
psnr_plots = sys.argv[5]

if parallel_network.lower() == 'True':
    parallel_network = True
elif parallel_network.lower() == 'False':
    parallel_network = False

if psnr_plots.lower() == 'True':
    psnr_plots = True
elif psnr_plots.lower() == 'False':
    psnr_plots = False
#ground_truth = utils.image_transform(
#    images_paths_list=[image_path],
#    transform=False,
#    seed=config_file["seed"]
#)[0]

def display_output(test_image, output, ground_truth, config_file, image_name):
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
    fig.savefig(os.path.join(config_file["save_default_path"], config_file["title"]) + "/plots/gaussian/" + image_name + ".png")


def evaluate(test_image_sequence, image_path, checkpoints_path, epoch, parallel_network, gaussian, test_gaussian_sequence):
    ground_truth = utils.image_transform(
        images_paths_list=[image_path],
        transform=False,
        seed=config_file["seed"]
    )[0]
    ground_truth = np.array(ground_truth, dtype='float64')

    for sequence, gaussian_sequence in zip(test_image_sequence, test_gaussian_sequence):
        test_image = utils.image_transform(
            images_paths_list=[image_path],
            gaussian_sequence=gaussian_sequence,
            particle_noise_density=sequence,
            transform=False,
            gaussian=gaussian,
            seed=config_file["seed"]
        )[0]

        width, height = test_image.size
        image_sample = ToTensor()(test_image).view(1,1,height,width).cuda()

        net = torch.load(checkpoints_path + "epoch-" + epoch + ".net")

        output = utils.predictMMSE(image_sample, 10, net, size=(height, width), parallel_network=parallel_network)
        display_output(test_image, output, ground_truth, config_file, str(sequence))
        
        if psnr_plots:
            psnr_data = psnr_plot(
                path_to_folder=pathlib.Path(checkpoints_path),
                image_sample=image_sample,
                height=height,
                width=width,
                gt=ground_truth,
            )

            np.save(plots_path + str(sequence) + "psnr_data.npy", psnr_data)


sequence = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
test_gaussian_sequence = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.09]
evaluate(sequence, image_path, checkpoints_path, epoch, parallel_network, gaussian, test_gaussian_sequence)

#test_image = utils.image_transform(
#    images_paths_list=image_path,
#    particle_noise_density=config_file["particle_noise_density"],
#    transform=True,
#    seed=config_file["seed"]
#)[0]

#width, height = test_image.size
#image_sample = ToTensor()(test_image).view(1,1,height,width).cuda()

#net = torch.load(checkpoints_path + "epoch-" + str(sys.argv[2]) + ".net")

#output = utils.predictMMSE(image_sample, 10, net, size=(height, width))
#psnr_data = psnr_plot(
#    image_sample=image_sample,
#    height=height,
#    width=width,
#    gt=ground_truth,
#)

#np.save(plots_path + "psnr_data.npy", psnr_data)

#plt.imsave(os.path.join(config_file["save_default_path"], config_file["title"]) + "/output_image.png", output, cmap='gray', dpi=300.0)
#plt.imsave(os.path.join(config_file["save_default_path"], config_file["title"]) + "/test_image.png", test_image, cmap='gray', dpi=300.0)
#plt.imsave(os.path.join(config_file["save_default_path"], config_file["title"]) + "/ground_truth_image.png", ground_truth, cmap='gray', dpi=300.0)





