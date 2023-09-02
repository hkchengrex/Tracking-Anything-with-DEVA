from typing import Dict, List
import torch

from deva.inference.object_info import ObjectInfo


class FrameInfo:
    def __init__(self, image: torch.Tensor, mask: torch.Tensor, segments_info: List[ObjectInfo],
                 ti: int, info: Dict):
        self.image = image
        self.mask = mask
        self.segments_info = segments_info
        self.ti = ti
        self.info = info

    @property
    def name(self):
        return self.info['frame'][0]

    @property
    def shape(self):
        return self.info['shape']

    @property
    def save_needed(self):
        return self.info['save'][0]
    
    @property
    def path_to_image(self):
        return self.info['path_to_image'][0]
