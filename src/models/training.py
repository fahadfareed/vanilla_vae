import torch.optim as optim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import time
import datetime
from src.features import training_features


def lossFunctionMSE(recon_x, x, gaussian_noise_std, data_std):
    b = (-0.5 * gaussian_noise_std)
    c = ((-0.5 / torch.exp(gaussian_noise_std)) * (x - recon_x) ** 2.0)
    reconstruction_error = -torch.mean(b+c)

    return reconstruction_error


def lossFunctionKLD(mu, sigma):
    kl_error = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return kl_error
    

def loss_fn(recon_x, x, mu, sigma, gaussian_noise_std, data_std):
    kl_loss = lossFunctionKLD(mu, sigma)
    
    reconstruction_loss = lossFunctionMSE(recon_x, x, gaussian_noise_std, data_std)
        
    return reconstruction_loss, kl_loss /float(x.numel())

def get_kl_weight(old_kl_weight,kl_annealtime):
    return min(old_kl_weight + (1./ kl_annealtime), 1.)


def trainNetwork(net, train_loader, val_loader, device,
                 data_mean,data_std,
                 model_name,
                 directory_path='.',
                 n_epochs=100,
                 batch_size=32,
                 lr=0.001,
                 val_loss_patience = 300,
                 kl_annealing = False,
                 kl_start = 2, # The number of epochs at which KL loss should be included
                 kl_annealtime = 5, # number of epochs over which KL scaling is increased from 0 to 1
                 gaussian_noise_std = None
                ):

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = training_features.LRScheduler(optimizer)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-12, verbose=True)
    early_stopping = training_features.EarlyStopping(patience=10)

    net.data_mean = torch.Tensor([data_mean]).to(device)
    net.data_std = torch.Tensor([data_std]).to(device)
    
    model_path = directory_path + "/" + "checkpoints/"
    plots_path = directory_path + "/" + "plots/"

    os.mkdir(model_path)
    os.mkdir(plots_path)
    assert(gaussian_noise_std is not None)
    assert(data_std is not None)
        
    old_kl_weight=0.0    

    seconds_last = time.time()
    for epoch in range(n_epochs):
        
        running_training_loss = []
        running_reconstruction_loss = []
        running_kl_loss = []
        if(kl_annealing==True):
            if(epoch>kl_start):
                new_kl_weight=get_kl_weight(old_kl_weight,kl_annealtime)
                old_kl_weight=new_kl_weight
            else:
                new_kl_weight=0.0
        else:
            new_kl_weight=1.0
        for x, _ in train_loader:
            x = x.cuda()
            x = (x-net.data_mean) / net.data_std
            #print(x.shape)
            mu, sigma = net.encoder(x)
            z = net.reparameterize(mu, sigma)
            recon, logvar_decoder = net.decoder(z)

            gaussian_noise_std = logvar_decoder

            reconstruction_loss, kl_loss = loss_fn(
                recon,
                x, 
                mu,
                sigma,
                gaussian_noise_std,
                net.data_std
            )
            if(kl_annealing==True):
                if(epoch>kl_start):
                    loss = reconstruction_loss+new_kl_weight*kl_loss
                else:
                    loss = reconstruction_loss
            else:
                loss = reconstruction_loss+kl_loss     
            running_training_loss.append(loss.item())
            running_reconstruction_loss.append(reconstruction_loss.item())
            running_kl_loss.append(kl_loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #net.eval() #Uncomment it if using batch norm/dropout
        
        ### Print training losses
        normalizer_train = len(train_loader.dataset)
        to_print = "Epoch[{}/{}] Training Loss: {:.3f} Reconstruction Loss: {:.3f} KL Loss: {:.3f}".format(epoch+1, 
                                    n_epochs, np.mean(running_training_loss), np.mean(running_reconstruction_loss), np.mean(running_kl_loss))
        print(to_print)
        print('saving',model_path+model_name+str(epoch)+".net")
        torch.save(net, model_path+model_name+str(epoch)+".net")
        
        ### Save training losses 

        if epoch == 0:
            loss_train_history = np.sum(running_training_loss)/normalizer_train
            reconstruction_loss_train_history = np.mean(running_reconstruction_loss)
            kl_loss_train_history = np.mean(running_kl_loss)
        else:
            loss_train_history = np.append(loss_train_history, np.mean(running_training_loss))
            reconstruction_loss_train_history = np.append(reconstruction_loss_train_history,
                                                          np.mean(running_reconstruction_loss))
            kl_loss_train_history = np.append(kl_loss_train_history, np.mean(running_kl_loss))
        
        np.save(plots_path+"train_loss.npy", np.array(loss_train_history))
        np.save(plots_path+"train_reco_loss.npy", np.array(reconstruction_loss_train_history))
        np.save(plots_path+"train_kl_loss.npy", np.array(kl_loss_train_history))

        ### Validation step
        running_validation_loss = []
        with torch.no_grad():
            for i, (x, _) in enumerate(val_loader):
                x = x.cuda()
                x = (x-net.data_mean) / net.data_std
                mu, sigma = net.encoder(x)
                z = net.reparameterize(mu, sigma)
                recon, logvar_decoder = net.decoder(z)

                gaussian_noise_std =logvar_decoder
                val_reconstruction_loss, val_kl_loss = loss_fn(
                    recon,
                    x,
                    mu,
                    sigma,
                    gaussian_noise_std, 
                    net.data_std,
                )
                val_loss = val_reconstruction_loss+val_kl_loss
                running_validation_loss.append(val_loss)

        normalizer_val = len(val_loader.dataset)
        total_epoch_loss_val = torch.mean(torch.stack(running_validation_loss))
        scheduler(total_epoch_loss_val)
        #early_stopping(total_epoch_loss_val)
        #if early_stopping.early_stop:
        #    break
        ### Save validation losses 
        if epoch == 0:
            loss_val_history = total_epoch_loss_val.item()
            patience_ = 0
        else:
            loss_val_history = np.append(loss_val_history, total_epoch_loss_val.item())

        np.save(plots_path+"val_loss.npy", np.array(loss_val_history))

        if total_epoch_loss_val.item() < 0.000001+np.min(loss_val_history):
            patience_ = 0

        else:
            patience_ +=1

        print("Patience:", patience_, "Validation Loss:", total_epoch_loss_val.item(), "Min validation loss:", np.min(loss_val_history))
        
        seconds=time.time()
        secondsElapsed=np.float(seconds-seconds_last)
        seconds_last=seconds
        remainingEps=n_epochs-(epoch+1)
        estRemainSeconds=(secondsElapsed)*(remainingEps)
        estRemainSecondsInt=int(secondsElapsed)*(remainingEps)
        print('Time for epoch: '+ str(int(secondsElapsed))+ 'seconds')
        print('Est remaining time: '+ str(datetime.timedelta(seconds=int(estRemainSeconds) )) +' or ' + str(int(estRemainSeconds))+ ' seconds')
        print('Est remaining time: '+ str(datetime.timedelta(seconds= estRemainSecondsInt)) +' or ' + str(estRemainSecondsInt)+ ' seconds')
        print("----------------------------------------", flush=True)
        
        
        if patience_ == val_loss_patience:
            break
            
    return loss_train_history, reconstruction_loss_train_history, kl_loss_train_history, loss_val_history
