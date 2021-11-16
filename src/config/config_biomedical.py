import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import data_parallel

from torch.utils.data import DataLoader

import os
import yaml
import sys
import argparse

from src.features import dataLoader as MyDataLoader
from src.features import utils
import pathlib
from tifffile import imread

from src.models import training, denoising_vanilla_vae
import matplotlib.pyplot as plt

# folder to load config file
CONFIG_PATH = "../../experiments/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config_file = yaml.safe_load(file)

    return config_file
my_parser = argparse.ArgumentParser(prog='Training', description='Start the training', allow_abbrev=False)
my_parser.add_argument('--experiment', action='store', type=str, required=True, help='enter the experiment number')
my_parser.add_argument('--multigpu', action='store', type=bool, required=True, help='enable multi gpu training')

args = my_parser.parse_args()

config_file = load_config(args.experiment + ".yaml")
parallel_network = args.multigpu
#config_file = load_config(str(sys.argv[1]) + ".yaml")
#parallel_network = sys.argv[2]
#if parallel_network.lower() == 'True':
#    parallel_network = True
#elif parallel_network.lower() == 'False':
#    parallel_network = False

observation= imread(config_file["data_directory"])
train_patches, val_patches = utils.get_trainval_patches(observation,augment=True,patch_size=128,num_patches=100)
x_train_tensor, x_val_tensor, data_mean, data_std = utils.preprocess(train_patches, val_patches,  config_file["if_array"])

print(x_train_tensor.shape)
print(x_val_tensor.shape)
print(data_mean)
print(data_std)
save_path = os.path.join(config_file["save_default_path"], config_file["title"])
os.mkdir(save_path)

for index, images in enumerate(train_patches[:10]):
    plt.imsave(os.path.join(config_file["save_default_path"], config_file["title"]) + "/" + str(index) + ".png", images, cmap='gray', dpi=300.0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model = Config(
    n_epochs=config_file["n_epochs"],
    batch_size=config_file["batch_size"],
    z_dim=config_file["z_dim"],
    x_train=x_train_tensor,
    x_val=x_val_tensor,
    learning_rate=config_file["learning_rate"],
    data_mean=data_mean.item(),
    data_std=data_std.item(),
    directory_path=save_path,
    bias=config_file["bias"],
    device=device,
    parallel_network=parallel_network,
)
yaml_path = save_path + "/config.yaml"
with open(yaml_path, 'w') as file:
    documents = yaml.dump(config_file, file)

vae_model.train()