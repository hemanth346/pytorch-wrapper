from __future__ import print_function
# from typing import Tuple, Callable

import numpy as np
import torch.nn.functional as F
import torch
import torchvision
# from .data_loader import CIFAR10Data

__all__ = ['UnNormalize', 'unnormalize_hwc_image', 'unnormalize_chw_image']



# def validate_loss(loss:str):
#     f_methods = dir(F)
#     f_methods = [method.lower() for method in f_methods]
#     try:
#         loss_idx = f_methods.index(loss.lower())
#     except:
#         raise ValueError('Invalid loss string input - must match pytorch function.')
#     return getattr(F, dir(F)[loss_idx])


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def unnormalize_hwc_image(image, mean, std):
    '''
    In torch Images are stored as shape [BATCH B, CHANNEL C, HEIGHT H, WIDTH W]
    As per discussions in pytorch forums, torch only supports NCHW format, can be changed using permute

    - Using permute method to swap the axis from HWC to CHW
    - Un-normalizes the image
    - Transpose/permute back CHW to HWC for imshow
    - Convert to np int array and multiple by 255

    :param image:
    :param mean:
    :param std:

    :return: unnormalized image as ndarray
    '''
    unorm = UnNormalize(mean=mean, std=std)
    # HWC to CHW and un-norm
    image = unorm(image.permute(2, 0, 1))
    # CHW to HWC for plots
    image = image.permute(1, 2, 0)
    return (image.numpy() * 255).astype(np.uint8)


def unnormalize_chw_image(image, mean, std):
    '''
    In torch Images are stored as shape [BATCH B, CHANNEL C, HEIGHT H, WIDTH W]

    - Un-normalizes the image
    - Transpose/permute back CHW to HWC for imshow
    - Convert to np int array and multiple by 255

    :param image:
    :param mean:
    :param std:

    :return: unnormalized image as ndarray
    '''
    unorm = UnNormalize(mean=mean, std=std)
    # CHW and un-norm
    image = unorm(image)
    # CHW to HWC for plots
    image = image.permute(1, 2, 0)
    return (image.numpy() * 255).astype(np.uint8)


