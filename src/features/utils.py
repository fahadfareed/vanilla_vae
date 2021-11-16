# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from os import pathconf
import time
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
import PIL
from sklearn.feature_extraction import image

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pathlib
import skimage
from matplotlib.image import imread


class Noising:
    def __init__(
        self,
        gaussian=False,
        particle_noise=False,
        seed=None,
        clip=False,
        gaussian_mean=0.0,
        gaussian_variance=0.05,
        particle_noise_density=1.0,
        padding=None,
        if_array=False,
        create_patches=True,
        patch_size=None,
    ) -> None:
        """
        Add noise to the image using this function. It requires
        noise weights to quantify the noise density in the image.
        Two types of noises are added here. 1) Gaussian, 2) Dot
        noise. 

        Parameters
        ----------
        images_list: list
            List of images to be noised.
        gaussian: boolean
            Set to true for adding gaussian noise.
        particle_noise: boolean
            Set to true for adding dot noise.
        seed: float
            Seed value for sampling.
        clip: boolean
            Set to true in order to clip the noise values
            in gaussian distribution to only positive values.
        gaussian_mean: float
            value of mean for the gaussian distribution to 
            sample from.
        gaussian_variance: float
            value of variance for the gaussian distribution to 
            sample from.
        """
        self.gaussian = gaussian
        self.particle_noise = particle_noise
        self.seed = seed
        self.clip = clip
        self.gaussian_mean = gaussian_mean
        self.gaussian_variance = gaussian_variance
        self.particle_noise_density = particle_noise_density
        self.padding = padding
        self.if_array = if_array
        self.patch_size = patch_size
        self.images = None
        self.image_tensors = None
        self.create_patches = create_patches
    
    def image_transform(self, images_list):
        images = []
        for image in images_list:
            if self.particle_noise:
                image = self.add_noise(image, self.particle_noise_density)
            elif self.gaussian:
                image = skimage.util.random_noise(
                        image,
                        mode='gaussian',
                        clip=self.clip,
                        var=self.gaussian_variance
                    )
            else:
                pass
            images.append(image)

        self.images = images
    
    def load_image_tensors(self):

        image_tensors = []
        if self.create_patches:
            image_tensors_list = [
                self.patching(self.convertNumpyToTensor(image).unsqueeze(0), self.patch_size, self.padding)
                for image in self.images
            ]
        else:
            image_tensors_list = [self.convertNumpyToTensor(image).unsqueeze(0) for image in self.images]

        image_tensors = torch.stack(image_tensors_list)
        image_tensors = image_tensors.view(-1, 1, image_tensors.size(2), image_tensors.size(3))

        if self.if_array:
            image_tensors = image_tensors.cpu().detach().numpy()
        
        self.image_tensors = image_tensors

    def convertNumpyToTensor(self, numpy_array):
        """Convert numpy array to PyTorch tensor. 
        Parameters
        ----------
        numpy_array: numpy array
            Numpy array.
        """
        return torch.from_numpy(numpy_array).float()
    
    def add_noise(self, image, particledensity):
        image_height, image_width = image.shape
        number = int(particledensity * image_width * image_height)
        top = np.random.uniform(low=0, high=(image_height-1), size=number)
        left = np.random.uniform(low=0, high=(image_width-1), size=number)
        for i in range(number):
            image[int(top[i])][int(left[i])] = 0.0
        
        return image

    def patching(self, image_tensor, patch_size, padding):
        """
        Create patches from the image tensor provided with a patch size.
        If the patch size does not quantify the image into equal size of
        tiles, padding should be set to true.

        Parameters
        ----------
        image_tensor: tensor
            Tensor containing image data.
        patch_size: tuple
            height and width of the patch required.
        padding: boolean
            Set to true if padding is needed on the image.
        """
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

    def noising(self, images_list):
        self.image_transform(images_list)
        self.load_image_tensors()
        return self.image_tensors

def load_paths(data_directory: pathlib.Path, n_samples: int):

    images_path_list = sorted(data_directory.rglob("*.png"))
    images_path_list = images_path_list[0:n_samples]

    return images_path_list


def load_images(data_directory, n_samples):
    """
    Function to simply load the image arrays from the paths provided.

    Parameters
    ----------
    data_directory: pathlib.Path
        Path to data directory.
    """
    train_images_folder_path = data_directory / "train"
    val_images_folder_path = data_directory / "val"
    train_images_path_list = load_paths(train_images_folder_path, n_samples)
    val_images_path_list = load_paths(val_images_folder_path, int(n_samples/10))
    train_images_list = [(imread(str(image_path))*255).astype("uint8") for image_path in train_images_path_list]
    val_images_list = [(imread(str(image_path))*255).astype("uint8") for image_path in val_images_path_list]
    return train_images_list, val_images_list
    

def convertToFloat32(train_images,val_images):
    """
    Converts the data to float 32 bit type.
 
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

def getMeanStdData(train_images, val_images, if_array=False):
    """
    Compute mean and standrad deviation of data.

    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
    """
    if if_array:
        x_train_, x_val_ = convertToFloat32(train_images, val_images)
        data = np.concatenate((x_train_,x_val_), axis=0)
        mean, std = np.mean(data), np.std(data)
    
    else:
        data = torch.cat((train_images, val_images))
        mean, std = data.mean(), data.std()

    return mean, std


def predictMMSE(image, samples, vae, size, parallel_network=False): 
    vae.eval()
    akku = np.zeros(size)
    if not parallel_network:
        mu, sigma = vae.encode(image)
    for i in range(samples):
        if parallel_network:
            mu, sigma, recon, _ = vae(image)
        else:
            z = vae.reparameterize(mu, sigma)
            recon = vae.decode(z)
        recon_cpu = recon.cpu()
        recon_numpy = recon_cpu.detach().numpy()
        recon_numpy.shape=size 
        akku+=recon_numpy
    output=akku/float(samples)
    return output


def getSamples(vae, size, zSize, mu=None, logvar=None, samples=1, tq=False):    
    """
    Generate synthetic samples from DivNoising network.

    Parameters
    ----------
    vae: VAE Object
        DivNoising model.
    size: tuple
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
        mu=torch.zeros(1,zSize,size[0],size[0]).cuda()
    if logvar is None:    
        logvar=torch.zeros(1,zSize,size[0],size[0]).cuda()

    results=[]
    for i in tqdm(range(samples),disable= not tq):
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
        recon_cpu = recon.cpu()
        recon_numpy = recon_cpu.detach().numpy()
        recon_numpy.shape=(recon_numpy.shape[-2],recon_numpy.shape[-1]) 
        results.append(recon_numpy)
    return np.array(results)

#def load_full_images(image_folder_path, n_samples, if_array=False):
#    images_path_list = load_paths(
#        data_directory=image_folder_path,
#        n_samples=n_samples
#    )

#    images_tensor = torch.stack([convertNumpyToTensor((imread(str(image_path))*255).astype("uint8")) for image_path in images_path_list])
    
#    full_images = images_tensor.view(-1, 1, images_tensor.size(2), images_tensor.size(3))

#    if if_array:
#        full_images = full_images.cpu().detach().numpy()

#    return full_images

def recreate_full_image(patches, samples):
    image_patches_list = []
    for sample in samples:
        image_patches_list.append(np.concatenate(patches[sample*samples:(sample+1)*samples], axis=1))
    full_image = np.concatenate(image_patches_list, axis=0)

    return full_image




