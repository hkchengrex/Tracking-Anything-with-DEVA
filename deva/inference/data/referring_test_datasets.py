import os
from os import path
import json
from collections import defaultdict
import numpy as np

from deva.inference.data.video_reader import VideoReader


class ReferringDAVISTestDataset:
    def __init__(self, image_dir, mask_dir, size=-1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size

        self.vid_list = sorted(os.listdir(self.mask_dir))

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
        )

    def get_scores(self, video):
        with open(path.join(self.mask_dir, video, 'scores.csv')) as f:
            lines = f.read().splitlines()
        scores = defaultdict(dict)
        for l in lines:
            frame, obj, score = l.split(',')
            scores[frame[:-4]][obj] = float(score)

        average_scores = {}
        for frame, all_objects in scores.items():
            average_scores[frame] = np.array(list(all_objects.values())).mean()

        return average_scores

    def __len__(self):
        return len(self.vid_list)


class ReferringYouTubeVOSTestDataset:
    def __init__(self, image_dir, mask_dir, json_dir, size=-1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size

        self.vid_list = sorted(os.listdir(self.mask_dir))
        self.req_frame_list = {}

        with open(json_dir) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                req_frames.extend(meta[vid]['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def get_videos(self):
        return self.vid_list

    def get_objects(self, video):
        return [
            obj for obj in sorted(os.listdir(path.join(self.mask_dir, video))) if '.csv' not in obj
        ]

    def _get_to_save_list(self, video, object_name):
        return self.req_frame_list[video]

    def get_offline_sampled_frames(self, video, object_name, num_sampled_frames):
        return VideoReader(
            video,
            path.join(self.image_dir, video),
            path.join(self.mask_dir, video),
            size=self.size,
            soft_mask=True,
            num_sampled_frames=num_sampled_frames,
            use_all_masks=True,
            to_save=self._get_to_save_list(video, object_name),
            object_name=object_name,
            enabled_frame_list=self._get_enabled_frame_list(video, object_name),
        )

    def get_partial_video_loader(self, video, object_name, *, start, end, reverse):
        return VideoReader(
            video,
            path.join(self.image_dir, video),
            path.join(self.mask_dir, video),
            size=self.size,
            soft_mask=True,
            start=start,
            end=end,
            reverse=reverse,
            to_save=self._get_to_save_list(video, object_name),
            object_name=object_name,
            enabled_frame_list=self._get_enabled_frame_list(video, object_name),
        )

    def get_scores(self, video):
        with open(path.join(self.mask_dir, video, 'scores.csv')) as f:
            lines = f.read().splitlines()
        scores = defaultdict(dict)
        enabled_frame_list = self._get_enabled_frame_list(video, None)
        for l in lines:
            frame, obj, score = l.split(',')
            if enabled_frame_list is not None and frame[:-4] not in enabled_frame_list:
                continue
            scores[obj][frame[:-4]] = float(score)
        return scores

    def _get_enabled_frame_list(self, video, object_name):
        # None -> enable all
        return None

    def __len__(self):
        return len(self.vid_list)
