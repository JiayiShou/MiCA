import os
import torch
import torchvisionI I
import torch.optim as optim
import matplotlib.pyplot as plt


from torchvision.utils import save_image
from PIL import Image

import numpy as np

def train(model, trainloader, epochs):
    train_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        for data in trainloader:
            img, _ = data
            img = img.cuda()
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

def test_image_reconstruction(model, testloader, imgName):
     for batch in testloader:
        img, _ = batch
        img = img.cuda()
        img = img.view(img.size(0), -1)
        outputs = model(img)
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        #plt.imshow(img)
        save_image(outputs, imgName)
        break
