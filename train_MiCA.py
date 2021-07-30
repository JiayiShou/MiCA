import os
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import save_image
from PIL import Image
from util import *

def train(model, trainloader, epochs, device, optimizer, criterion):
    train_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        for data in trainloader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, epochs, loss))
        if epoch % 5 == 0:
            save_decoded_image(outputs.cpu().data, epoch)
    return train_loss

def train_vae(model, trainloader, epochs, device, optimizer, criterion):
    train_loss = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in trainloader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(data)
            mse_loss = criterion(reconstruction, data)
            fi_loss = final_loss(mse_loss, mu, logvar)
            running_loss += fi_loss.item()
            fi_loss.backward()
            optimizer.step()

        loss = running_loss/len(trainloader.dataset)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, epochs, loss))
        if epoch % 5 == 0:
            save_decoded_image(reconstruction.cpu().data, epoch)
    return train_loss

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def test_image_reconstruction(model, testloader, imgName, device):
     for batch in testloader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = model(img)
        if len(outputs) != 1:
            recon = outputs(0)
        else:
            outputs = outputs
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        #plt.imshow(img)
        save_image(outputs, imgName)
        break
