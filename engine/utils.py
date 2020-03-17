from __future__ import print_function
# from typing import Tuple, Callable

import numpy as np
import torch.nn.functional as F
import torch
import torchvision
from .data_loader import DataLoader

# __all__ = ['validate_loss', 'UnNormalize']
__all__ = ['UnNormalize', 'unnormalize']


def unnormalize(img):
    img = img.numpy().astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * DataLoader.cifar10_std[i]) + DataLoader.cifar10_mean[i]

    return np.transpose(img, (1, 2, 0))


# def validate_loss(loss:str):
#     f_methods = dir(F)
#     f_methods = [method.lower() for method in f_methods]
#     try:
#         loss_idx = f_methods.index(loss.lower())
#     except:
#         raise ValueError('Invalid loss string input - must match pytorch function.')
#     return getattr(F, dir(F)[loss_idx])




# class UnNormalize(object):
#     # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/11
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#             # The normalize code -> t.sub_(m).div_(s)
#         return t

class UnNormalize(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
