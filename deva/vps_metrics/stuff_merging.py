import os
from os import path

import json
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from functools import partial
from progressbar import progressbar

from deva.utils.vipseg_categories import VIPSEG_CATEGORIES
from deva.utils.pano_utils import IDPostprocessor, id_to_rgb
from multiprocessing import Pool
"""
Post-processing is done online, so technically it can be run with the original evaluation. 
But that introduces additional complexities that are just tailored for VPS/VIPSeg. 
So I make this part into a post-processing script.

Specifically, it does the following:
1. For every "thing" segment, whenever its category changes, we give it a new object id.
    This is because the evaluation script assumes that objects with the same object id 
    have the same category. This change aligns with the original formula in 
    Video Panoptic Segmentation, CVPR 2020.
2. It stores a mapping table from "stuff" category to object id. Whenever we encounter a segment
    with a "stuff" category, we apply this mapping.
"""


def process_single_video(vid_ann, input_path, output_path):
    video_id = vid_ann['video_id']
    video_output_annotation = []
    video_output = {'video_id': video_id, 'annotations': video_output_annotation}
    output_path = path.join(output_path, 'pan_pred', video_id)
    os.makedirs(output_path, exist_ok=True)

    converter = IDPostprocessor()

    for ann in vid_ann['annotations']:
        file_name = ann['file_name']
        segments_info = ann['segments_info']
        output_segments_info = []
        output_annotation = {'file_name': ann['file_name'], 'segments_info': output_segments_info}
        video_output_annotation.append(output_annotation)

        mask = np.array(
            Image.open(
                path.join(input_path, 'pan_pred', video_id,
                          file_name.replace('.jpg', '.png')))).astype(np.int32)
        mask = mask[:, :, 0] + mask[:, :, 1] * 256 + mask[:, :, 2] * 256 * 256
        output_mask = np.zeros_like(mask)

        for segment in segments_info:
            id = segment['id']
            category_id = segment['category_id']
            isthing = vipseg_cat_to_isthing[category_id]
            new_id = converter.convert(id, category_id, isthing)
            output_mask[mask == id] = new_id

            if isthing:
                # is not stuff
                output_segment = {
                    'id': new_id,
                    'category_id': segment['category_id'],
                    'isthing': 1,
                }
                output_segments_info.append(output_segment)

        # a pass for the merged stuff objects
        for cat, new_id in converter.stuff_to_id.items():
            area = int((output_mask == new_id).sum())
            assert not vipseg_cat_to_isthing[cat]
            if area > 0:
                output_segment = {
                    'id': new_id,
                    'category_id': cat,
                    'isthing': 0,
                }
                output_segments_info.append(output_segment)

        # save the new output mask
        output_mask = id_to_rgb(output_mask)
        output_mask = Image.fromarray(output_mask)
        output_mask.save(path.join(output_path, file_name.replace('.jpg', '.png')))

    return video_output


vipseg_cat_to_isthing = {d['id']: d['isthing'] == 1 for d in VIPSEG_CATEGORIES}


def merge_stuff(input_path, output_path):
    with open(path.join(input_path, 'pred.json')) as f:
        annotations = json.load(f)['annotations']

    output_annotations = []
    pool = Pool(16)
    for out_vid_ann in progressbar(pool.imap(
            partial(process_single_video, input_path=input_path, output_path=output_path),
            annotations),
                                   max_value=len(annotations)):
        output_annotations.append(out_vid_ann)

    output_json = {'annotations': output_annotations}
    with open(path.join(output_path, 'pred.json'), 'w') as f:
        json.dump(output_json, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    merge_stuff(input_path, output_path)
