from PIL import Image
import os
from os import path
import sys
import numpy as np
import tqdm

from deva.utils.palette import davis_palette


def limit_max_id(input_path, output_path, max_num_objects=20):
    videos = sorted(os.listdir(input_path))
    for video in tqdm.tqdm(videos):
        existing_objects = []

        video_path = path.join(input_path, video)
        frames = sorted(os.listdir(video_path))

        # determine the objects to keep
        for frame in frames:
            mask = Image.open(path.join(video_path, frame))
            mask = np.array(mask).astype(np.int32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 2]
            labels = np.unique(mask)
            labels = labels[labels != 0]
            labels_area = [np.sum(mask == label) for label in labels]

            labels_sorted_by_area = [x for _, x in sorted(zip(labels_area, labels), reverse=True)]
            if len(labels_sorted_by_area) + len(existing_objects) <= max_num_objects:
                existing_objects += labels_sorted_by_area
            else:
                existing_objects += labels_sorted_by_area[:max_num_objects - len(existing_objects)]

            if len(existing_objects) == max_num_objects:
                break

        assert len(existing_objects) <= max_num_objects

        # remove the objects that are not in the existing_objects list
        for frame in frames:
            mask = Image.open(path.join(video_path, frame))
            mask = np.array(mask).astype(np.int32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 2]
            labels = np.unique(mask)
            labels = labels[labels != 0]

            new_mask = np.zeros_like(mask, dtype=np.uint8)
            for new_idx, label in enumerate(existing_objects):
                new_mask[mask == label] = new_idx + 1

            mask = Image.fromarray(new_mask)
            mask.putpalette(davis_palette)
            os.makedirs(path.join(output_path, video), exist_ok=True)
            mask.save(path.join(output_path, video, frame))


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    limit_max_id(input_path, output_path)