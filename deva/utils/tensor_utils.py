from typing import List, Iterable
import torch
import torch.nn.functional as F


# STM
def pad_divide_by(in_img: torch.Tensor, d: int) -> (torch.Tensor, Iterable[int]):
    h, w = in_img.shape[-2:]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array


def unpad(img: torch.Tensor, pad: Iterable[int]) -> torch.Tensor:
    if len(img.shape) == 4:
        if pad[2] + pad[3] > 0:
            img = img[:, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, :, pad[0]:-pad[1]]
    elif len(img.shape) == 3:
        if pad[2] + pad[3] > 0:
            img = img[:, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, pad[0]:-pad[1]]
    elif len(img.shape) == 5:
        if pad[2] + pad[3] > 0:
            img = img[:, :, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, :, :, pad[0]:-pad[1]]
    elif len(img.shape) == 2:
        if pad[2] + pad[3] > 0:
            img = img[pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, pad[0]:-pad[1]]
    else:
        raise NotImplementedError
    return img
