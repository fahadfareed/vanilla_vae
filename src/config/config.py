import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import os
import yaml
import sys

from src.features import dataLoader as MyDataLoader
from src.features import utils
import pathlib

from src.models import training, denoising_vanilla_vae
import matplotlib.pyplot as plt

class Config:
    def __init__(
        self,
        n_epochs,
        batch_size,
        z_dim,
        x_train,
        x_val,
        learning_rate,
        data_mean,
        data_std,
        directory_path,
        bias,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.x_train = x_train
        self.x_val = x_val
        self.data_mean = data_mean
        self.data_std = data_std
        self.learning_rate = learning_rate
        self.val_loss_patience = 100
        self.in_channels = 1
        self.init_filters = 32
        self.n_filters_per_depth = 2
        self.n_depth = 2
        self.gaussian_noise_std = 1.0
        self.device = torch.device("cuda:0")
        self.kl_annealing = True
        self.kl_start = 0
        self.kl_annealtime = 3
        self.directory_path = directory_path
        self.bias = bias


    def train(self):
        vae = denoising_vanilla_vae.VAE(
            z_dim=self.z_dim,
            in_channels=self.in_channels,
            init_filters=self.init_filters,
            n_filters_per_depth=self.n_filters_per_depth,
            n_depth=self.n_depth,
            bias=self.bias,
        )

        train_dataset = MyDataLoader.MyDataset(self.x_train,self.x_train)
        val_dataset = MyDataLoader.MyDataset(self.x_val,self.x_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        model_name = "epoch-"

        trainHist, reconHistory, klHist, valHist = training.trainNetwork(
            net=vae,
            train_loader=train_loader, 
            val_loader=val_loader,
            device=self.device,
            directory_path=self.directory_path,
            model_name=model_name,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            val_loss_patience=self.val_loss_patience,
            kl_annealing=self.kl_annealing,
            kl_start=self.kl_start, 
            kl_annealtime=self.kl_annealtime,
            data_mean=self.data_mean,
            data_std=self.data_std, 
            gaussian_noise_std=self.gaussian_noise_std
        )


# folder to load config file
CONFIG_PATH = "../../experiments/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config_file = yaml.safe_load(file)

    return config_file

config_file = load_config(str(sys.argv[1]) + ".yaml")

train_images_folder_path = pathlib.Path(config_file["data_directory"]) / "train"
val_images_folder_path = pathlib.Path(config_file["data_directory"]) / "val"

train_images_path_list = utils.load_paths(
    data_directory=train_images_folder_path,
    n_samples=config_file["n_samples"],
)

val_images_path_list = utils.load_paths(
    data_directory=val_images_folder_path,
    n_samples=int(config_file["n_samples"]/10),
)

train_images = utils.image_transform(
    images_paths_list=train_images_path_list,
    seed=config_file["seed"],
    particle_noise_density=config_file["particle_noise_density"],
    transform=config_file["transform"],
    gaussian=config_file["gaussian"],
    clip=config_file["clip"],
)
val_images = utils.image_transform(
    images_paths_list=val_images_path_list,
    seed=config_file["seed"],
    particle_noise_density=config_file["particle_noise_density"],
    transform=config_file["transform"],
    gaussian=config_file["gaussian"],
    clip=config_file["clip"],
)

x_train_tensor = utils.load_image_patches(
    images=train_images,
    patch_size=config_file["patch_size"],
    padding=config_file["padding"],
    if_array=config_file["if_array"]
)

x_val_tensor = utils.load_image_patches(
    images=val_images,
    patch_size=config_file["patch_size"],
    padding=config_file["padding"],
    if_array=config_file["if_array"]
)

mean, std = utils.getMeanStdData(x_train_tensor, x_val_tensor)
print(x_train_tensor.shape)
print(x_val_tensor.shape)
print(mean)
print(std)
save_path = os.path.join(config_file["save_default_path"], config_file["title"])
os.mkdir(save_path)

for index, images in enumerate(train_images[:10]):
    plt.imsave(os.path.join(config_file["save_default_path"], config_file["title"]) + "/" + str(index) + ".png", images, cmap='gray', dpi=300.0)

vae_model = Config(
    n_epochs=config_file["n_epochs"],
    batch_size=config_file["batch_size"],
    z_dim=config_file["z_dim"],
    x_train=x_train_tensor,
    x_val=x_val_tensor,
    learning_rate=config_file["learning_rate"],
    data_mean=mean.item(),
    data_std=std.item(),
    directory_path=save_path,
    bias=config_file["bias"]
)
yaml_path = save_path + "_config.yaml"
with open(yaml_path, 'w') as file:
    documents = yaml.dump(config_file, file)

vae_model.train()