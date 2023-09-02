from typing import Dict, List, Optional
import os
from os import path
import copy

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import pycocotools.mask as mask_utils

from deva.dataset.utils import im_normalization


class VideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(
        self,
        vid_name,
        image_dir,
        mask_dir,
        *,
        size=-1,
        to_save=None,
        use_all_masks=False,
        size_dir=None,
        start=-1,
        end=-1,
        num_sampled_frames=-1,
        reverse=False,
        soft_mask=False,
        object_name=None,
        multi_object=True,
        segmentation_from_dict: Optional[Dict[str, Dict]] = None,
        enabled_frame_list: Optional[List[str]] = None,
    ):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        use_all_masks - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        soft_mask - read (from sub-folders) and return soft probability mask
        object_name - if none, read from all objects. if not none, read that object only. 
                        only valid in soft mask mode
        segmentation_from_dict - if not None, read segmentation from this dictionary instead
        """
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_masks
        self.soft_mask = soft_mask
        self.object_name = object_name
        self.multi_object = multi_object
        self.segmentation_from_dict = segmentation_from_dict
        self.enabled_frame_list = enabled_frame_list
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        if segmentation_from_dict is None:
            # read all frames in the image directory
            self.frames = sorted(os.listdir(self.image_dir))
        else:
            # read frames from the dictionary
            first_frame = sorted(os.listdir(self.image_dir))[0]
            extension = first_frame[-4:]
            self.frames = sorted(segmentation_from_dict.keys())
            # add extensions -- pretty dumb but simple
            self.frames = [f + extension for f in self.frames]

        if enabled_frame_list is not None:
            self.frames = [f for f in self.frames if f[:-4] in enabled_frame_list]

        # enforce start and end frame if needed
        self._all_frames = copy.deepcopy(self.frames)
        if start >= 0:
            if end >= 0:
                self.frames = self.frames[start:end]
            else:
                self.frames = self.frames[start:]
        elif end >= 0:
            self.frames = self.frames[:end]

        if num_sampled_frames > 0:
            # https://stackoverflow.com/a/9873804/3237438
            assert start < 0 and end < 0
            m = num_sampled_frames
            n = len(self.frames)
            m = min(m, n)
            indices = [i * n // m + n // (2 * m) for i in range(m)]
            self.frames = [self.frames[i] for i in indices]

        if reverse:
            self.frames = list(reversed(self.frames))

        if self.segmentation_from_dict is not None:
            # decoding masks from the dict
            self.palette = None
            self.first_mask_frame = self.frames[0]
        elif soft_mask:
            # reading probability masks
            self.palette = None
            if multi_object:
                if object_name is not None:
                    # pick one of many objects, soft mask
                    self.mask_dir = path.join(self.mask_dir, object_name)
                    self.first_mask_frame = sorted(os.listdir(self.mask_dir))[0]
                else:
                    # use all objects, soft mask
                    self.prob_folders = sorted(os.listdir(self.mask_dir))
                    self.prob_folders = [f for f in self.prob_folders if '.csv' not in f]
                    self.first_mask_frame = sorted(
                        os.listdir(path.join(self.mask_dir, self.prob_folders[0])))[0]
            else:
                # single object soft mask
                self.mask_dir = path.join(self.mask_dir)
                self.first_mask_frame = sorted(os.listdir(self.mask_dir))[0]
        else:
            # reading ID masks with palette
            self.palette = Image.open(path.join(mask_dir,
                                                sorted(os.listdir(mask_dir))[0])).getpalette()
            self.first_mask_frame = sorted(os.listdir(self.mask_dir))[0]

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
            if self.soft_mask:
                self.mask_transform = transforms.Compose([
                    transforms.Resize(size,
                                      interpolation=InterpolationMode.BILINEAR,
                                      antialias=True),
                ])
            else:
                self.mask_transform = transforms.Compose([
                    transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
                ])
        self.size = size

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
        img = self.im_transform(img)

        load_mask = self.use_all_mask or (frame[:-4] == self.first_mask_frame[:-4])

        if load_mask:
            if self.segmentation_from_dict is not None:
                # decoding masks from the dict
                pred = self.segmentation_from_dict[frame[:-4]][self.object_name]
                mask = mask_utils.decode(pred['segmentation'])
                mask = self.mask_transform(mask)
                all_masks = torch.FloatTensor(np.array(mask)).unsqueeze(0)
                valid_labels = torch.LongTensor(list(range(1, 2)))

            elif self.soft_mask:
                all_masks = []
                if self.object_name is not None or not self.multi_object:
                    # pick one of many objects, soft mask
                    # or, single object, soft mask
                    mask_path = path.join(self.mask_dir, frame[:-4] + '.png')
                    mask = Image.open(mask_path)
                    mask = self.mask_transform(mask)
                    mask = torch.FloatTensor(np.array(mask)) / 255
                    all_masks.append(mask)
                    if self.object_name is not None:
                        info['object_name'] = self.object_name
                elif self.multi_object:
                    # use all objects, soft mask
                    for prob_folder in self.prob_folders:
                        mask_path = path.join(self.mask_dir, prob_folder, frame[:-4] + '.png')
                        assert path.exists(mask_path)

                        mask = Image.open(mask_path)
                        mask = self.mask_transform(mask)
                        mask = torch.FloatTensor(np.array(mask)) / 255
                        all_masks.append(mask)
                all_masks = torch.stack(all_masks, dim=0)
                valid_labels = torch.LongTensor(list(range(1, len(all_masks) + 1)))
            else:
                mask_path = path.join(self.mask_dir, frame[:-4] + '.png')
                if path.exists(mask_path):
                    mask = Image.open(mask_path).convert('P')
                    mask = self.mask_transform(mask)
                    mask = torch.LongTensor(np.array(mask))
                    valid_labels = torch.unique(mask)
                    valid_labels = valid_labels[valid_labels != 0]
                    all_masks = mask
                else:
                    all_masks = valid_labels = None

            if all_masks is not None:
                data['mask'] = all_masks
                data['valid_labels'] = valid_labels

        info['shape'] = shape
        info['need_resize'] = not (self.size < 0)
        info['time_index'] = self._all_frames.index(frame)
        info['path_to_image'] = im_path
        data['rgb'] = img
        data['info'] = info

        return data

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)