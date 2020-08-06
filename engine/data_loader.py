import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt

from .utils import UnNormalize, unnormalize_chw_image, unnormalize_hwc_image

# __all__ = ['CIFAR10Data', 'CIFAR10Data.get_loaders', 'CIFAR10Data.mean', 'CIFAR10Data.std', 'CIFAR10Data.classes']


class AlbumentationToPytorchTransforms():
    """
    Helper class to convert Albumentations compose
    into compatible for pytorch transform compose
    """

    def __init__(self, albumentation_compose=None):
        self.transform = albumentation_compose

    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)['image']


class VisualizeLoader(object):
    '''
    Show data from loaders
    '''

    def __init__(self, loader, labels, mean, std):
        self.loader = loader
        self.targets = labels
        self.mean = mean
        self.std = std
        self.__images = None
        self.__labels = None

    def show_images(self, number=5, ncols=5, figsize=(15, 10), iterate=True):

        if iterate:# or not images:
            self.__images, self.__labels = next(iter(self.loader))
            # images, labels = self.__images, self.__labels

        images, labels = self.__images, self.__labels
        # selecting random sample of number
        img_list = random.sample(range(1, images.shape[0]), number)

        self.set_plt_param()
        rows = (number//ncols) + 1
        axes = self.make_grid(rows, ncols, figsize)
        for idx, label in enumerate(img_list):
            img = unnormalize_chw_image(images[label], self.mean, self.std)
            axes[idx].imshow(img, interpolation='bilinear')
            axes[idx].set_title(self.targets[labels[label]])
        # Hide empty subplot boundaries
        [ax.set_visible(False) for ax in axes[idx + 1:]]
        plt.show()

    @staticmethod
    def make_grid(nrows, ncols=3, figsize=(6.0, 4.0)):
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        axes = ax.flatten()
        return axes
        pass

    @staticmethod
    def set_plt_param():
        rc = {"axes.spines.left": False,
              "axes.spines.right": False,
              "axes.spines.bottom": False,
              "axes.spines.top": False,
              "axes.grid": False,
              "xtick.bottom": False,
              "xtick.labelbottom": False,
              "ytick.labelleft": False,
              "ytick.left": False}
        plt.rcParams.update(rc)


    def __call__(self, number=5, *args, **kwargs):
        return self.show_images(number=number, *args, **kwargs)


class CIFAR10Data():
    # CIFAR10Data():

    mean = (0.491, 0.482, 0.447)
    std = (0.247, 0.243, 0.262)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    @staticmethod
    def get_loaders(batch_size:int=64, augmentations=[], seed:int=None, shuffle=True, num_workers=4, pin_memory=True):
        '''
        Load CIFAR10 data

        :params: 
            batch_size : int
            augmentation transforms : List
            seed for dataloaders: int
        :returns: 
            train dataloader
            test dataloader
        '''

        transforms_list = [ 
            transforms.ToTensor(), 
            transforms.Normalize(CIFAR10Data.mean, CIFAR10Data.std)
            ]

        if augmentations:
            transforms_list = [augmentations] + transforms_list

        train_transforms = transforms.Compose(transforms_list)
        test_transforms = transforms.Compose(transforms_list)

        #Get the Train and Test Set
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

        # CUDA?
        cuda = torch.cuda.is_available()
        if seed:
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)
        print('Using cuda') if cuda else print('Using cpu')
        dataloader_args = {
            'shuffle': shuffle, 
            'batch_size': batch_size, 
            'num_workers': num_workers, 
            'pin_memory': pin_memory
            } if cuda else {
            'shuffle': shuffle, 
            'batch_size': batch_size
            }

        train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

        return train_loader, test_loader

    # @staticmethod
    # showimages