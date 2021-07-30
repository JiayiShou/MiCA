import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image


def save_decoded_image(img, epoch):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, './FashionMNIST_Images/image{}.png'.format(epoch))


class add_gaussian_noise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        chance = (torch.rand(tensor.size())<0.25).int()
        return tensor + chance * torch.rand(tensor.size()) + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def matplotlib_imshow(img, one_channel=False):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plotLoss(train_loss, model_name):
    title_name = model_name + ' Train Loss'
    file_name = model_name + '_fashionmnist_loss.png'
    plt.figure()
    plt.plot(train_loss)
    plt.title(title_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(file_name)

def make_folder(folder_name):
    current_directory = os.getcwd()
    directory = folder_name
    if not os.path.exists(directory):
    	os.mkdir(os.path.join(current_directory, directory))

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
