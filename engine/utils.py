from __future__ import print_function
# from typing import Tuple, Callable

import torch.nn.functional as F
import torch as t

# __all__ = ['validate_loss', 'UnNormalize']
__all__ = ['UnNormalize']


# def validate_loss(loss:str):
#     f_methods = dir(F)
#     f_methods = [method.lower() for method in f_methods]
#     try:
#         loss_idx = f_methods.index(loss.lower())
#     except:
#         raise ValueError('Invalid loss string input - must match pytorch function.')
#     return getattr(F, dir(F)[loss_idx])




class UnNormalize(object):
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/11
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
            # The normalize code -> t.sub_(m).div_(s)
        return t