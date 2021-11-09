"Visualization functions to plot the results."

import matplotlib.pyplot as plt
import numpy as np


def plot_one_experiment(directory_path, noise_index_list):

    fig1 = plt.figure(figsize=(30,20))
    ax1 = fig1.add_subplot(111)
    #ax1.set_yticks(np.arange(14.0, 17.0, .1))
    for i in noise_index_list:
        data = np.load(directory_path + str(i) + "psnr_data.npy")
        ax1.plot(data,label=i)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("PSNR")
    colormap = plt.cm.RdYlBu #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])
    ax1.legend()

def multiple_plots(directory_path, experiment_sequence, z_dimension_sequence, nrows, ncols):
    x_coordinate = [i for i in range(0, ncols) for _ in range(ncols)]
    y_coordinate = [i for _ in range(ncols) for i in range(0, nrows)]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(40,20))
    for h, i, j, k in zip(x_coordinate, y_coordinate, experiment_sequence, z_dimension_sequence):
        plot_path = directory_path + j + "/plots/"
        data = np.load(plot_path + "1.0psnr_data.npy")
        ax[h, i].plot(data)
        ax[h, i].title.set_text("z-dimensions=" + k)
        ax[h, i].set_yticks(np.arange(13.0, 20.0, .5))
        ax[h, i].set_xlabel("Epochs")
        ax[h, i].set_ylabel("PSNR")
    plt.show()