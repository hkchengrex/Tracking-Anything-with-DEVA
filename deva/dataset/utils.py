import random
import numpy as np
import torch
import torchvision.transforms as transforms

im_mean = (124, 116, 104)

im_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inv_im_trans = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])


def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for ni, l in enumerate(labels):
        Ms[ni] = (masks == l).astype(np.uint8)

    return Ms
