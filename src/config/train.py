import pathlib
from sklearn.feature_extraction import image
import yaml
import os
import torch
import sys
import numpy as np

from src import features
from DivNoising.divnoising import utils

from src.config import config
import matplotlib.pyplot as plt
from tifffile import imread

# folder to load config file
CONFIG_PATH = "/home/fahad/master_thesis/vanilla_vae/experiments/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config_file = yaml.safe_load(file)

    return config_file


#my_parser = argparse.ArgumentParser(prog='Training', description='Start the training', allow_abbrev=False)
#my_parser.add_argument('--experiment', action='store', type=str, required=True, help='enter the experiment number')
#my_parser.add_argument('--multigpu', dest='multigpu', action='store_true', help='enable multi gpu training')
#my_parser.add_argument('--no-multigpu', dest='multigpu', action='store_false', help='disable multi gpu training')
#my_parser.set_defaults(multigpu=False)
#my_parser.add_argument('--synthetic', dest='synthetic', action='store_true', help='training includes synthetic data')
#my_parser.add_argument('--no-synthetic', dest='synthetic', action='store_false', help='training includes biomedical data')
#my_parser.set_defaults(biomedical=False)

#args = my_parser.parse_args()

#config_file = load_config(args.experiment + ".yaml")
#parallel_network = args.multigpu

def pre_processing(config_name, dataset_type):
    config_file = load_config(config_name + ".yaml")

    if dataset_type=="synthetic":
        data_directory = pathlib.Path(config_file["data_directory"])

        train_images_list, val_images_list = features.utils.load_images(
            data_directory=data_directory,
            n_samples=config_file["n_samples"]
        )

        image_transform = features.utils.Noising(
            gaussian=config_file["gaussian"],
            particle_noise=config_file["transform"],
            seed=config_file["seed"],
            clip=config_file["clip"],
            particle_noise_density=config_file["particle_noise_density"],
            patch_size=config_file["patch_size"],
            padding=config_file["padding"],
            if_array=config_file["if_array"],
            create_patches=config_file["patches"],
        )

        x_train_tensor = image_transform.noising(images_list=train_images_list)
        x_val_tensor = image_transform.noising(images_list=val_images_list)

        mean, std = features.utils.getMeanStdData(x_train_tensor, x_val_tensor)
        print(x_train_tensor.shape)
        print(x_val_tensor.shape)
        print(mean)
        print(std)
        
    elif dataset_type=="biomedical":
        observation= imread(config_file["data_directory"])
        train_patches, val_patches = utils.get_trainval_patches(observation,augment=True,patch_size=128,num_patches=100)
        x_train_tensor, x_val_tensor, mean, std = utils.preprocess(train_patches, val_patches)

    else:
        print("data type not recognized")
        return None

    #save_path = os.path.join(config_file["save_default_path"], config_file["title"])
    #os.mkdir(save_path)

    #for index, images in enumerate(np.array(x_train_tensor[:10])):
    #    plt.imsave(
    #        os.path.join(config_file["save_default_path"],
    #        config_file["title"]) + "/" + str(index) + ".png",
    #        images[0], cmap='gray',
    #        dpi=300.0
    #    )

    return x_train_tensor, x_val_tensor, mean, std

def model_train(x_train_tensor, x_val_tensor, mean, std, config_name, gpu_ids, dataset_type):
    config_file = load_config(config_name + ".yaml")
    if dataset_type=="synthetic":
        if len(gpu_ids)>1:
            parallel_model = True
        else:
            parallel_model = False
    if dataset_type=="biomedical":
        if len(gpu_ids)>1:
            print("only a single gpu is allowed for training")
            return None
        else:
            parallel_model = False
    
    save_path = os.path.join(config_file["save_default_path"], config_file["title"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_model = config.Config(
        n_epochs=config_file["n_epochs"],
        batch_size=config_file["batch_size"],
        z_dim=config_file["z_dim"],
        x_train=x_train_tensor,
        x_val=x_val_tensor,
        learning_rate=config_file["learning_rate"],
        data_mean=mean.item(),
        data_std=std.item(),
        directory_path=save_path,
        bias=config_file["bias"],
        device=device,
        parallel_network=parallel_model,
    )

    yaml_path = save_path + "/config.yaml"

    with open(yaml_path, 'w') as file:
        documents = yaml.dump(config_file, file)

    vae_model.train()
