import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from vanilla_vae.features import dataLoader as MyDataLoader
from vanilla_vae.models import training, denoising_vanilla_vae


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
        device,
        parallel_network=False,
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
        self.device = device
        self.kl_annealing = True
        self.kl_start = 0
        self.kl_annealtime = 3
        self.directory_path = directory_path
        self.bias = bias
        self.parallel_network = parallel_network


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
        if self.parallel_network:
            vae = nn.DataParallel(vae)
            vae = vae.to(self.device)
    
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
            gaussian_noise_std=self.gaussian_noise_std,
            parallel_network=self.parallel_network
        )


