from os import pathconf
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
import pathlib
import skimage
from imageaugment import augment


GAMMA=(.8, 1.0)
ANGLE_FINAL=(0, 0)
ANGLE_TRANSIENT=(0, 0)
SHIFT=(0, 0)
SCALE=(1.0, 1.0)
THRESHOLD=(.65, .80)
BRIGHTNESS=(1.0, 1.3)
DITHERPROB=0.0
FLIPPROB=0.0
VLINEPROB=.5
MAXVLINES=2
LINEWIDTH=(0.001, 0.002)
PARTICLEDENSITY=(.001, .01)
PARTICLESIZE=(.0001, .001)

def image_transform(
    images_paths_list,
    seed,
    particle_noise_density=1.0,
    transform=True,
    gaussian=False,
    clip=False,
):

    particle_density = tuple([particle_noise_density*value for value in PARTICLEDENSITY])
    image_transform = augment.get_random_faxify(
        gamma=GAMMA,
        angle_final=ANGLE_FINAL,
        angle_transient=ANGLE_TRANSIENT,
        shift=SHIFT,
        scale=SCALE,
        threshold=THRESHOLD,
        brightness=BRIGHTNESS,
        ditherprob=DITHERPROB,
        flipprob=FLIPPROB,
        vlineprob=VLINEPROB,
        maxvlines=MAXVLINES,
        linewidth=LINEWIDTH,
        particledensity=particle_density,
        particlesize=PARTICLESIZE,
        seed=seed,
    )
    images = []
    for image_path in images_paths_list:
        image = PIL.Image.open(image_path).convert("L")
        if transform:
            image = image_transform(image)
        if gaussian:
            image = PIL.Image.fromarray(
                (skimage.util.random_noise(np.array(image),
                mode='gaussian',
                seed=seed,
                clip=clip,
                var=.05))*255
            ).convert("L")
        images.append(image)

    return images

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

def getMeanStdData(train_images,val_images, if_array=False):
    """Compute mean and standrad deviation of data. 
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
    """
    if not if_array:
        train_images = train_images.cpu().detach().numpy()*255
        val_images = val_images.cpu().detach().numpy()*255

    x_train_ = np.squeeze(train_images, axis=1)
    x_val_ = np.squeeze(val_images, axis=1)
    data = np.concatenate((x_train_,x_val_), axis=0)
    mean, std = np.mean(data), np.std(data)

    return mean/255.0, std/255.0

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
    return output*255


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

def load_paths(data_directory: pathlib.Path, n_samples: int):

    images_path_list = sorted(data_directory.rglob("*.png"))
    images_path_list = images_path_list[0:n_samples]

    return images_path_list

def patching(image_tensor, patch_size, padding):
    height, width  = patch_size
    if padding:
        w_pad1 = (width - (image_tensor.size(2)%width)) // 2
        w_pad2 = (width - (image_tensor.size(2)%width)) - w_pad1
        h_pad1 = (height - (image_tensor.size(1)%height)) // 2
        h_pad2 = (height - (image_tensor.size(1)%height)) - h_pad1
        image_tensor = F.pad(image_tensor, (w_pad1, w_pad2, h_pad1, h_pad2), value=1)
    patches = image_tensor.unfold(1, height, height).unfold(2, width, width)
    patches = patches.contiguous().view(-1, height, width)
    
    return patches

def load_image_patches(
    images,
    patch_size,
    padding=False,
    if_array=False
):

    patched_images = [
        patching(ToTensor()(image), patch_size, padding)
        for image in images
    ]

    images_tensor = torch.stack(patched_images)
    patched_images = images_tensor.view(-1, 1, images_tensor.size(2), images_tensor.size(3))

    if if_array:
        patched_images = patched_images.cpu().detach().numpy()*255
    
    return patched_images

def load_full_images(image_folder_path, n_samples, if_array=False):
    images_path_list = load_paths(
        data_directory=image_folder_path,
        n_samples=n_samples
    )

    images_tensor = torch.stack([ToTensor()(PIL.Image.open(image_path)) for image_path in images_path_list])
    
    full_images = images_tensor.view(-1, 1, images_tensor.size(2), images_tensor.size(3))

    if if_array:
        full_images = full_images.cpu().detach().numpy()*255

    return full_images


def recreate_full_image(patches, samples):
    image_patches_list = []
    for sample in samples:
        image_patches_list.append(np.concatenate(patches[sample*samples:(sample+1)*samples], axis=1))
    full_image = np.concatenate(image_patches_list, axis=0)

    return full_image

