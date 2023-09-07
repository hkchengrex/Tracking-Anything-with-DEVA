import os
from os import path
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np


class SimpleVideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    This simple version:
    1. Does not load the mask/json
    2. Does not normalize the input
    3. Does not resize
    """
    def __init__(
        self,
        image_dir,
    ):
        """
        image_dir - points to a directory of jpg images
        """
        self.image_dir = image_dir
        self.frames = sorted(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        frame = self.frames[idx]

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')
        img = np.array(img)

        return img, im_path
    
    def __len__(self):
        return len(self.frames)
    

def no_collate(x):
    return x