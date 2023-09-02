import json
import sys
import os
from os import path
import tqdm

gt_json_path = sys.argv[1]
pred_path = sys.argv[2]
out_path = sys.argv[3]

with open(gt_json_path) as f:
    json_file = json.load(f)

for sequence in tqdm.tqdm(json_file['sequences']):
    dataset = sequence['dataset']
    seq_name = sequence['seq_name']

    sequence['segmentations'] = []

    with open(path.join(pred_path, dataset, seq_name, 'pred.json')) as f:
        pred_json = json.load(f)
        track_category_id = {}
        for frame_segmentation in pred_json['segmentations']:
            this_frame_segmentation = {}

            for segmentation_dict in frame_segmentation['segmentations']:
                this_frame_segmentation[segmentation_dict['id']] = {
                    'rle': segmentation_dict['rle']['counts']
                }
                track_category_id[segmentation_dict['id']] = 0
            sequence['segmentations'].append(this_frame_segmentation)

        sequence['track_category_ids'] = track_category_id


with open(out_path, 'w') as f:
    json.dump(json_file, f)
