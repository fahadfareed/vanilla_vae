"""Command line interface for Training models"""

import click
import pathlib
import os
from src.config import train as set_config
from src.config import eval as evaluation
from src.features import utils
from tifffile import imread

@click.group()
def entry_point():
    """Entry point for the training cli."""


@entry_point.command()
@click.argument(
    "config_name",
    type=str,
    #help="""""",
)
@click.option(
    "--gpu-ids",
    default="0",
    help="""List of gpu ids to be used for training. One GPU --gpu-ids=n (where n is the 
    id of the desired GPU to be used), and multiple GPUs --gpu-ids=l,m,n (where l,m,n are 
    the ids of the desired GPUs to be used). If not given, the default value is 0.""",
    required=False,
)
@click.option(
    "--dataset-type",
    default="synthetic",
    help="""For data pre-processing it is important to whether the data is synthetic or 
    biomedical due to difference in the patching criteria. For synthetic dataset enter 
    'synthetic' as an input, else 'biomedical'.""",
    required=False,
)
def train(
    config_name: str,
    gpu_ids: str,
    dataset_type: str,
):
    """
    Enter the configuration file name in order to let the training parameters be 
    #set through the variables mentioned in the configuration file.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    x_train_tensor, x_val_tensor, mean, std = set_config.pre_processing(
        config_name=config_name,
        dataset_type=dataset_type
    )

    set_config.model_train(
        x_train_tensor=x_train_tensor,
        x_val_tensor=x_val_tensor,
        mean=mean,
        std=std,
        config_name=config_name,
        gpu_ids=gpu_ids,
        dataset_type=dataset_type
    )


@entry_point.command()
@click.argument(
    "config_name",
    type=str,
    #help="""""",
)
@click.option(
    "--epoch",
    default="199",
    help="""Enter the epoch number of the model which has to be evaluated.""",
    required=False,
)
@click.option(
    "--multi-gpu",
    default=False,
    help="""Bool variable that should either be set to True or False. If the model is 
    trained using parallel mode, it can only be evaluate with the same method. If not 
    given, the default value is set to False.""",
    required=False,
)
@click.option(
    "--dataset-type",
    default="synthetic",
    help="""For data pre-processing it is important to whether the data is synthetic or 
    biomedical due to difference in the patching criteria. For synthetic dataset enter 
    'synthetic' as an input, else 'biomedical'.""",
    required=False,
)
@click.option(
    "--plots",
    default=False,
    help="""For training & predictions, plots can be created of training losses & PSNR
    values for which the bool variable should either be set to True or False.""",
    required=False,
)
@click.option(
    "--noise-type",
    default="dot_noise",
    help="""For data pre-processing it is important to whether the data is synthetic or 
    biomedical due to difference in the patching criteria. For synthetic dataset enter 
    'synthetic' as an input, else 'biomedical'.""",
    required=False,
)
def evaluate(
    config_name: str,
    multi_gpu: click.BOOL,
    epoch: str,
    dataset_type: str,
    plots: click.BOOL,
    noise_type: str
):
    """
    Enter the configuration file name in order to let the training parameters be 
    #set through the variables mentioned in the configuration file.
    """
    x_train_tensor, x_val_tensor, mean, std = evaluation.eval(
        config_name=config_name,
        multi_gpu=multi_gpu,
        epoch=epoch,
        dataset_type=dataset_type,
        plots=plots,
        noise_type=noise_type,
    )
