import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms #
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

#from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

import numpy as np
from util import *
from dataset import *
from Autoencoders import *
from train_MiCA import *
#Add use of GPU, test run on one autoencoder at a time.
#validation loss??

make_folder("FashionMNIST_Images")

epochs = 3
lr_rate = 1e-3
batch_size = 128
device = get_device()
print(device)

transform_normal = transforms.Compose([transforms.ToTensor()])
transform_denoised = transforms.Compose([transforms.ToTensor(), add_gaussian_noise(0., 1.)])

trainloader_normal, testloader_normal = create_dataloader(transform_normal, batch_size, transform_normal, transform_normal)
trainloader_denoised, testloader_denoised = create_dataloader(transform_denoised, batch_size, transform_denoised, transform_denoised)
'''
#stacked autoencoder
automodel_stacked = StackedAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(automodel_stacked.parameters(), lr=lr_rate)
train_loss_stacked = train(automodel_stacked, trainloader_normal, epochs, device, optimizer, criterion)

# Specify a path and save
automodel_stacked_PATH = "automodel_stacked.pt"
torch.save(automodel_stacked.state_dict(), automodel_stacked_PATH)

# Load
#automodel_stacked = torch.load(automodel_stacked_PATH)
#automodel_stacked.eval()

plotLoss(train_loss_stacked, 'stacked_ae')
test_image_reconstruction(automodel_stacked, testloader_normal, 'stacked_ae_fashionmnist_recon.png',device)

#denoised autoencoder
automodel_denoised = StackedAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(automodel_denoised.parameters(), lr=lr_rate)
train_loss_denoised = train(automodel_denoised, trainloader_denoised, epochs, device, optimizer, criterion)
automodel_denoised_PATH = "automodel_stacked.pt"
torch.save(automodel_denoised.state_dict(), automodel_denoised_PATH)

# Load
#automodel_denoised = torch.load(automodel_denoised_PATH)
#automodel_denoised.eval()
plotLoss(train_loss_denoised, 'denoised_ae')
test_image_reconstruction(automodel_denoised, testloader_denoised, 'denoised_ae_fashionmnist_recon.png',device)
'''
#variational autoencoder
automodel_variational = LinearVAE().to(device)
criterion = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(automodel_variational.parameters(), lr=lr_rate)
train_loss_vatiational = train_vae(automodel_variational, trainloader_normal, epochs, device, optimizer, criterion)
automodel_variational_PATH = "automodel_stacked.pt"
torch.save(automodel_variational.state_dict(), automodel_variational_PATH)

# Load
#automodel_variational= torch.load(automodel_variational_PATH)
#automodel_variational.eval()
test_image_reconstruction(automodel_variational, testloader_normal, 'variational_ae_fashionmnist_recon.png',device)
