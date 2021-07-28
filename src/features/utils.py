import time
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
import PIL

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def convertToFloat32(train_images,val_images):
    """Converts the data to float 32 bit type. 
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
        """
    x_train = train_images.astype('float32')
    x_val = val_images.astype('float32')
    return x_train, x_val

def getMeanStdData(train_images,val_images):
    """Compute mean and standrad deviation of data. 
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
    """
    x_train_ = train_images.astype('float32')
    x_val_ = val_images.astype('float32')
    data = np.concatenate((x_train_,x_val_), axis=0)
    mean, std = np.mean(data), np.std(data)
    return mean, std

def convertNumpyToTensor(numpy_array):
    """Convert numpy array to PyTorch tensor. 
    Parameters
    ----------
    numpy_array: numpy array
        Numpy array.
    """
    return torch.from_numpy(numpy_array)

def predictMMSE(image, samples, vae, size): 
    vae.eval()
    mu, sigma = vae.encode(image)
    akku = np.zeros(size)
    for i in range(samples):
        z = vae.reparameterize(mu, sigma)
        recon = vae.decode(z)
        recon_cpu = recon.cpu()
        recon_numpy = recon_cpu.detach().numpy()
        recon_numpy.shape=size 
        akku+=recon_numpy
    output=akku/float(samples)
    return output


def getSamples(vae, size, zSize, mu=None, logvar=None, samples=1, tq=False):    
    """Generate synthetic samples from DivNoising network. 
    Parameters
    ----------
    vae: VAE Object
        DivNoising model.
    size: int
        Size of generated image in the bottleneck.
    zSize: int
        Dimension of latent space for each pixel in bottleneck.
    mu: PyTorch tensor
        latent space mean tensor.
    logvar: PyTorch tensor
        latent space log variance tensor.
    samples: int
        Number of synthetic samples to generate.
    tq: boolean
        If tqdm should be active or not to indicate progress.
    """
    if mu is None:
        mu=torch.zeros(1,zSize,size,size).cuda()
    if logvar is None:    
        logvar=torch.zeros(1,zSize,size,size).cuda()

    results=[]
    for i in tqdm(range(samples),disable= not tq):
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
        recon_cpu = recon.cpu()
        recon_numpy = recon_cpu.detach().numpy()
        recon_numpy.shape=(recon_numpy.shape[-2],recon_numpy.shape[-1]) 
        results.append(recon_numpy) 
    return np.array(results)


def load_image_array_patches(image_folder_path, n_samples):
    train_image_folder_path = image_folder_path / "train"
    val_image_folder_path = image_folder_path / "val"
    train_image_path_list = sorted(train_image_folder_path.rglob("*.png"))
    val_image_path_list = sorted(val_image_folder_path.rglob("*.png"))

    train_image_path_list = train_image_path_list[0:n_samples]
    val_image_path_list = val_image_path_list[0:int(n_samples/10)]
    
    train_patched_images = []
    val_patched_images = []
    for image_path in train_image_path_list:
        image = PIL.Image.open(image_path)
        x = ToTensor()(image)

        kh, kw = 192, 128  # kernel size
        dh, dw = 192, 128 # stride

        patches = x.unfold(1, kh, dh).unfold(2, kw, dw)
        patches = patches.contiguous().view(-1, kh, kw)
        patches = np.array(patches)
        train_patched_images.append(patches)
    
    for image_path in val_image_path_list:
        image = PIL.Image.open(image_path)
        x = ToTensor()(image)

        kh, kw = 192, 128  # kernel size
        dh, dw = 192, 128 # stride

        patches = x.unfold(1, kh, dh).unfold(2, kw, dw)
        patches = patches.contiguous().view(-1, kh, kw)
        patches = np.array(patches)
        val_patched_images.append(patches)

    train_patched_images_array = np.expand_dims(np.concatenate(np.array(train_patched_images)*255, axis=0), axis=3)
    val_patched_images_array = np.expand_dims(np.concatenate(np.array(val_patched_images)*255, axis=0), axis=3)
    
    return train_patched_images_array, val_patched_images_array


def load_image_tensor_patches(image_folder_path, n_samples, size=(1536,1024), padding=False, resize=False):
    train_image_folder_path = image_folder_path / "train"
    val_image_folder_path = image_folder_path / "val"
    train_image_path_list = sorted(train_image_folder_path.rglob("*.png"))
    val_image_path_list = sorted(val_image_folder_path.rglob("*.png"))

    train_image_path_list = train_image_path_list[0:n_samples]
    val_image_path_list = val_image_path_list[0:int(n_samples/10)]
    
    train_patched_images = []
    val_patched_images = []

    for image_path in train_image_path_list:
        image = PIL.Image.open(image_path)
        if resize:
            image = image.resize(size)
        image = image.convert("L")
        x = ToTensor()(image)

        kh, kw = 192, 128  # kernel size
        dh, dw = 192, 128 # stride

        if padding:
            w_pad1 = (kw - (x.size(2)%kw)) // 2
            w_pad2 = (kw - (x.size(2)%kw)) - w_pad1
            h_pad1 = (kh - (x.size(1)%kh)) // 2
            h_pad2 = (kh - (x.size(1)%kh)) - h_pad1
            x = F.pad(x, (w_pad1, w_pad2, h_pad1, h_pad2), value=1)

        patches = x.unfold(1, kh, dh).unfold(2, kw, dw)
        patches = patches.contiguous().view(-1, kh, kw)
        train_patched_images.append(patches)
    
    for image_path in val_image_path_list:
        image = PIL.Image.open(image_path)
        if resize:
            image = image.resize(size)
        image = image.convert("L")
        x = ToTensor()(image)

        kh, kw = 192, 128  # kernel size
        dh, dw = 192, 128 # stride

        if padding:
            w_pad1 = (kw - (x.size(2)%kw)) // 2
            w_pad2 = (kw - (x.size(2)%kw)) - w_pad1
            h_pad1 = (kh - (x.size(1)%kh)) // 2
            h_pad2 = (kh - (x.size(1)%kh)) - h_pad1
            x = F.pad(x, (w_pad1, w_pad2, h_pad1, h_pad2), value=1)

        patches = x.unfold(1, kh, dh).unfold(2, kw, dw)
        patches = patches.contiguous().view(-1, kh, kw)
        val_patched_images.append(patches)

    train_images_tensor = torch.stack(train_patched_images)
    val_images_tensor = torch.stack(val_patched_images)
    train_images_tensor = train_images_tensor.view(-1, 1, train_images_tensor.size(2), train_images_tensor.size(3))
    val_images_tensor = val_images_tensor.view(-1, 1, val_images_tensor.size(2), val_images_tensor.size(3))
    train_patched_images.extend(val_patched_images)
    mean = torch.mean(torch.cat(train_patched_images))
    std = torch.std(torch.cat(train_patched_images))

    return train_images_tensor, val_images_tensor, mean, std


def load_full_image_arrays(image_folder_path, n_samples):
    train_image_folder_path = image_folder_path / "train"
    val_image_folder_path = image_folder_path / "val"
    train_image_path_list = sorted(train_image_folder_path.rglob("*.png"))
    val_image_path_list = sorted(val_image_folder_path.rglob("*.png"))
    
    train_image_path_list = train_image_path_list[0:n_samples]
    val_image_path_list = val_image_path_list[0:int(n_samples/10)]
    
    train_images = [np.array((PIL.Image.open(image_path))) for image_path in train_image_path_list]
    val_images = [np.array((PIL.Image.open(image_path))) for image_path in val_image_path_list]
    train_images_array = np.expand_dims(np.array(train_images), axis=3)
    val_images_array = np.expand_dims(np.array(val_images), axis=3)

    return train_images_array, val_images_array

def load_full_image_tensors(image_folder_path, n_samples):
    train_image_folder_path = image_folder_path / "train"
    val_image_folder_path = image_folder_path / "val"
    train_image_path_list = sorted(train_image_folder_path.rglob("*.png"))
    val_image_path_list = sorted(val_image_folder_path.rglob("*.png"))
    train_image_path_list = train_image_path_list[0:n_samples]
    val_image_path_list = val_image_path_list[0:int(n_samples/10)]
    train_images = [ToTensor()(PIL.Image.open(image_path)) for image_path in train_image_path_list]
    val_images = [ToTensor()(PIL.Image.open(image_path)) for image_path in val_image_path_list]

    train_images_tensor = torch.stack(train_images)
    val_images_tensor = torch.stack(val_images)
    train_images_tensor = train_images_tensor.view(-1, 1, train_images_tensor.size(2), train_images_tensor.size(3))
    val_images_tensor = val_images_tensor.view(-1, 1, val_images_tensor.size(2), val_images_tensor.size(3))
    
    train_images.extend(val_images)
    mean = torch.mean(torch.cat(train_images))
    std = torch.std(torch.cat(train_images))

    return train_images_tensor, val_images_tensor, mean, std