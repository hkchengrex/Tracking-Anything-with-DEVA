import os
from os import path

from deva.inference.data.video_reader import VideoReader


class DAVISSaliencyTestDataset:
    def __init__(self, image_dir, mask_dir, imset=None, size=-1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size

        if imset is None:
            self.vid_list = sorted(os.listdir(self.mask_dir))
        else:
            with open(imset) as f:
                self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(
                video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                to_save=[name[:-4] for name in os.listdir(path.join(self.mask_dir, video))],
                size=self.size,
                soft_mask=True,
                use_all_masks=True,
                multi_object=False,
            )

    def get_videos(self):
        return self.vid_list

    def get_offline_sampled_frames(self, video, num_sampled_frames):
        return VideoReader(
            video,
            path.join(self.image_dir, video),
            path.join(self.mask_dir, video),
            to_save=[name[:-4] for name in os.listdir(path.join(self.mask_dir, video))],
            size=self.size,
            soft_mask=True,
            num_sampled_frames=num_sampled_frames,
            use_all_masks=True,
            multi_object=False,
        )

    def get_partial_video_loader(self, video, *, start, end, reverse):
        return VideoReader(
            video,
            path.join(self.image_dir, video),
            path.join(self.mask_dir, video),
            to_save=[name[:-4] for name in os.listdir(path.join(self.mask_dir, video))],
            size=self.size,
            soft_mask=True,
            start=start,
            end=end,
            reverse=reverse,
            multi_object=False,
        )

    def __len__(self):
        return len(self.vid_list)