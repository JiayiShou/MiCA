import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


class ThreeCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        x3 = self.base_transform(x)
        return [x1, x2, x3]

def get_dataset_stat(dataset):

    if dataset == 'FashionMNIST':
        image_size = 28  #desired size after block. Single int means square with side length x.
        mean = [0.5, 0.5, 0.5] #temporary data
        std = [0.2, 0.2, 0.2] #temporary data
        n_class = 10
    return image_size, mean, std, n_class


def create_dataset(dataset, train_transform, test_transform):
    print("Create dataset with tripple transform")
    train_transform = ThreeCropsTransform(train_transform)
    test_transform = ThreeCropsTransform(test_transform)

    if dataset == 'FashionMNIST':
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform = train_transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform = test_transform)

    return trainset, testset

#takes FashionMNIST and put out dataloader according to the transform.
def create_dataloader(dataset, batch_size, train_transform, test_transform):

    trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform = train_transform)
    testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform = test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader

class ImageFolderTripple(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """
    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderTripple, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img2 = self.transform(image)
        img3 = self.transform(image)

        return [img, img2, img3], target
