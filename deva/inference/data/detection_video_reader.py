import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import numpy as np

from deva.dataset.utils import im_normalization


class DetectionVideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self,
                 vid_name,
                 image_dir,
                 mask_dir,
                 size=-1,
                 to_save=None,
                 size_dir=None,
                 start=-1,
                 end=-1,
                 reverse=False):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        """
        # TODO: determine if_rgb automatically
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        self.frames = sorted(os.listdir(self.image_dir))
        if start > 0:
            self.frames = self.frames[start:]
        if end > 0:
            self.frames = self.frames[:end]
        if reverse:
            self.frames = reversed(self.frames)

        self.palette = Image.open(path.join(mask_dir, self.frames[0].replace('.jpg',
                                                                             '.png'))).getpalette()
        self.first_gt_path = path.join(self.mask_dir, self.frames[0].replace('.jpg', '.png'))

        if size < 0:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])
            self.mask_transform = transforms.Compose([])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
            ])
        self.size = size
        self.is_rgb = None

    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        mask_path = path.join(self.mask_dir, frame[:-4] + '.png')
        img = self.im_transform(img)

        if path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = self.mask_transform(mask)
            if mask.mode == 'RGB':
                mask = np.array(mask, dtype=np.int32)
                mask = mask[:, :, 0] + mask[:, :, 1] * 256 + mask[:, :, 2] * 256 * 256
                self.is_rgb = True
            else:
                mask = mask.convert('P')
                mask = np.array(mask, dtype=np.int32)
                self.is_rgb = False
            data['mask'] = mask

        # defer json loading to the model
        json_path = path.join(self.mask_dir, frame[:-4] + '.json')
        if path.exists(json_path):
            info['json'] = json_path

        info['is_rgb'] = self.is_rgb
        info['shape'] = shape
        info['need_resize'] = not (self.size < 0)
        info['path_to_image'] = im_path
        data['rgb'] = img
        data['info'] = info

        return data

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)