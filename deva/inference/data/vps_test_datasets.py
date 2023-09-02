import os
from os import path
import json

from deva.inference.data.detection_video_reader import DetectionVideoReader


class VIPSegDetectionTestDataset:
    def __init__(self, image_dir, mask_dir, size=-1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.vid_list = sorted(os.listdir(self.mask_dir))
        self.vid_list = [v for v in self.vid_list if not v.endswith('.json')]

    def get_datasets(self):
        for video in self.vid_list:
            yield DetectionVideoReader(
                video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                to_save=[name[:-4] for name in os.listdir(path.join(self.mask_dir, video))],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class BURSTDetectionTestDataset:
    def __init__(self, image_dir, mask_dir, gt_json_dir, size=-1, *, start=None, count=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size

        # read the json file to get a list of videos and frames to save
        with open(gt_json_dir, 'r') as f:
            json_file = json.load(f)
            sequences = json_file['sequences']
            split = json_file['split']

        assert split == 'test' or split == 'val'

        # load a randomized ordering of BURST videos for a balanced load
        with open(f'./deva/utils/burst_{split}.txt', mode='r') as f:
            randomized_videos = list(f.read().splitlines())

        # subsample a list of videos for processing
        if start is not None and count is not None:
            randomized_videos = randomized_videos[start:start + count]
            print(f'Start: {start}, Count: {count}, End: {start+count}')

        self.vid_list = []
        self.frames_to_save = {}
        for sequence in sequences:
            dataset = sequence['dataset']
            seq_name = sequence['seq_name']
            video_name = path.join(dataset, seq_name)
            if video_name not in randomized_videos:
                continue
            self.vid_list.append(video_name)

            annotated_image_paths = sequence['annotated_image_paths']
            self.frames_to_save[video_name] = [p[:-4] for p in annotated_image_paths]
            assert path.exists(path.join(image_dir, video_name))
            assert path.exists(path.join(mask_dir, video_name))

        assert len(self.vid_list) == len(randomized_videos)
        # to use the random ordering
        self.vid_list = randomized_videos

        print(f'Actual total: {len(self.vid_list)}')

    def get_datasets(self):
        for video in self.vid_list:
            yield DetectionVideoReader(
                video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                to_save=self.frames_to_save[video],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)