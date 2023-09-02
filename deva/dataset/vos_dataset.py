import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from deva.dataset.utils import im_normalization, im_mean, reseed


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick num_frames frames
    - Pick max_num_obj objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is limited by max_jump
    """
    def __init__(self,
                 im_root,
                 gt_root,
                 max_jump,
                 *,
                 size=384,
                 subset=None,
                 num_frames=3,
                 max_num_obj=3,
                 data_ratio=1.0):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj
        self.size = size

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        self.videos = self.videos[:int(len(self.videos) * data_ratio)]
        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=25,
                                    shear=20,
                                    interpolation=InterpolationMode.BILINEAR,
                                    fill=im_mean),
            transforms.RandomResizedCrop((self.size, self.size),
                                         scale=(0.16, 2.0),
                                         interpolation=InterpolationMode.BILINEAR,
                                         antialias=True)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=25,
                                    shear=20,
                                    interpolation=InterpolationMode.NEAREST,
                                    fill=0),
            transforms.RandomResizedCrop((self.size, self.size),
                                         scale=(0.16, 2.0),
                                         interpolation=InterpolationMode.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = []  # Append with sampled frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(
                range(max(0, frames_idx[-1] - this_max_jump),
                      min(length, frames_idx[-1] + this_max_jump + 1))).difference(set(frames_idx))
            while (len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(
                    range(max(0, frames_idx[-1] - this_max_jump),
                          min(length, frames_idx[-1] + this_max_jump + 1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                this_im = self.pair_im_lone_transform(this_im)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels != 0]

            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))

        masks = np.stack(masks, 0)

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, self.size, self.size), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, self.size, self.size), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks == l)
            cls_gt[this_mask] = i + 1
            first_frame_gt[0, i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)