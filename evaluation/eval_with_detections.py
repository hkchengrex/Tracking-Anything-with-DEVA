import os
from os import path
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm
import json
from multiprocessing import Process
from functools import partial

from deva.inference.data.vps_test_datasets import VIPSegDetectionTestDataset, BURSTDetectionTestDataset
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.object_utils import convert_json_dict_to_objects_info
from deva.inference.frame_utils import FrameInfo
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config

from deva.vps_metrics.stuff_merging import merge_stuff
from deva.vps_metrics.eval_stq_vipseg import eval_stq
from deva.vps_metrics.eval_vpq_vipseg import eval_vpq
from deva.inference.postprocess_unsup_davis17 import limit_max_id

# for id2rgb
np.random.seed(42)
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--img_path', default='./example/vipseg')
parser.add_argument('--mask_path')
parser.add_argument('--json_path', default=None)
parser.add_argument('--detection_every', type=int, default=5)
parser.add_argument('--num_voting_frames',
                    default=3,
                    type=int,
                    help='Number of frames selected for voting. only valid in semionline')
parser.add_argument('--dataset', default='vipseg', help='vipseg/burst/unsup_davis17/demo')
parser.add_argument('--max_missed_detection_count', type=int, default=5)
# skip VPQ/STQ computation
parser.add_argument('--no_metrics', action='store_true')

parser.add_argument('--temporal_setting', default='semionline', help='semionline/online')
parser.add_argument('--max_num_objects',
                    default=-1,
                    type=int,
                    help='Max. num of objects to keep in memory. -1 for no limit')

# the options below are only valid for burst
parser.add_argument('--start', type=int, default=None, help='for distributed testing')
parser.add_argument('--count', type=int, default=None, help='for distributed testing')
parser.add_argument('--burst_gt_json', default='../BURST/val/all_classes.json')

# only valid for VIPSeg
parser.add_argument('--vipseg_root', default='../VIPSeg/VIPSeg_720P')

# this option is only valid for unsup_davis17; limit the maximum number of predicted objects
parser.add_argument('--postprocess_limit_max_id', type=int, default=20)

add_common_eval_args(parser)
network, config, args = get_model_and_config(parser)
"""
Temporal setting
"""
temporal_setting = args.temporal_setting.lower()
assert temporal_setting in ['semionline', 'online']
"""
Data preparation
"""
dataset_name = args.dataset.lower()
assert dataset_name in ['vipseg', 'burst', 'unsup_davis17',
                        'demo'], f'Unknown dataset {dataset_name}'
print(f'Dataset: {dataset_name}')
is_vipseg = (dataset_name == 'vipseg')
is_burst = (dataset_name == 'burst')
is_davis = (dataset_name == 'unsup_davis17')
is_demo = (dataset_name == 'demo')

# try to find json path is not given
if args.json_path is None:
    if path.exists(path.join(args.mask_path, 'pred.json')):
        args.json_path = path.join(args.mask_path, 'pred.json')
out_path = args.output

# try to find the real mask path if it is hidden behind pan_pred
if path.exists(path.join(args.mask_path, 'pan_pred')):
    args.mask_path = path.join(args.mask_path, 'pan_pred')
if is_vipseg or is_davis or is_demo:
    meta_dataset = VIPSegDetectionTestDataset(args.img_path, args.mask_path, args.size)
elif is_burst:
    meta_dataset = BURSTDetectionTestDataset(args.img_path,
                                             args.mask_path,
                                             args.burst_gt_json,
                                             args.size,
                                             start=args.start,
                                             count=args.count)
else:
    raise NotImplementedError

torch.autograd.set_grad_enabled(False)

# Set up loader
meta_loader = meta_dataset.get_datasets()
"""
Read the global pred.json if any
"""
global_json_enabled = args.json_path is not None
per_vid_json_enabled = None
if global_json_enabled:
    print(f'Using a global json file {args.json_path}')
    with open(args.json_path, 'r') as f:
        all_json_info = json.load(f)
    all_json_info = all_json_info['annotations']

    video_id_to_annotation = {}
    for ann in all_json_info:
        video_id_to_annotation[ann['video_id']] = ann['annotations']

if is_vipseg:
    # we will export this as a single json for VPQ/STQ evaluation
    output_json_annotations = []

total_process_time = 0
total_frames = 0

# Start eval
pbar = tqdm(meta_loader, total=len(meta_dataset))
for vid_reader in pbar:
    loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
    vid_name = vid_reader.vid_name
    pbar.set_description(vid_name)
    vid_length = len(loader)
    next_voting_frame = args.num_voting_frames - 1
    # no need to count usage for LT if the video is not that long anyway
    config['enable_long_term_count_usage'] = (
        config['enable_long_term']
        and (vid_length / (config['max_mid_term_frames'] - config['min_mid_term_frames']) *
             config['num_prototypes']) >= config['max_long_term_elements'])

    try:
        processor = DEVAInferenceCore(network, config=config)
        result_saver = ResultSaver(out_path,
                                   vid_name,
                                   dataset=dataset_name,
                                   palette=vid_reader.palette,
                                   object_manager=processor.object_manager)

        for ti, data in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=args.amp):
                image = data['rgb'].cuda()[0]
                mask = data.get('mask')
                if mask is not None:
                    mask = mask.cuda()[0]
                info = data['info']
                frame = info['frame'][0]
                shape = info['shape']
                need_resize = info['need_resize'][0]
                is_rgb = info['is_rgb'][0]
                path_to_image = info['path_to_image'][0]
                if args.save_all:
                    info['save'][0] = True
                if is_rgb:
                    # if the mask format is RGB (instead of grayscale/palette), we need
                    # more usable IDs (>255)
                    processor.enabled_long_id()

                segments_info = None
                if not global_json_enabled:
                    # safety check
                    json_path = info.get('json')
                    if per_vid_json_enabled is None:
                        if json_path is None:
                            print('Neither global nor per-video json exist.')
                            per_vid_json_enabled = False
                        else:
                            print('Using per-video json.')
                            per_vid_json_enabled = True
                    elif json_path is None and per_vid_json_enabled:
                        raise RuntimeError(
                            f'Per-video json is enabled but not found for {vid_name}.')

                    # read the per-video pred.json
                    if per_vid_json_enabled:
                        with open(json_path[0], 'r') as f:
                            segments_info = json.load(f)
                        processor.enabled_long_id()
                else:
                    # read from the global json
                    segments_info = video_id_to_annotation[vid_name][ti]['segments_info']
                    processor.enabled_long_id()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                segments_info = convert_json_dict_to_objects_info(mask,
                                                                  segments_info,
                                                                  dataset=dataset_name)
                frame_info = FrameInfo(image, mask, segments_info, ti, info)

                if temporal_setting == 'semionline':
                    if ti + args.num_voting_frames > next_voting_frame:
                        # wait for more frames before proceeding
                        processor.add_to_temporary_buffer(frame_info)

                        if ti == next_voting_frame:
                            # process this clip
                            this_image = processor.frame_buffer[0].image
                            this_ti = processor.frame_buffer[0].ti
                            this_frame_name = processor.frame_buffer[0].name
                            save_this_frame = processor.frame_buffer[0].save_needed
                            path_to_image = processor.frame_buffer[0].path_to_image

                            _, mask, new_segments_info = processor.vote_in_temporary_buffer(
                                keyframe_selection='first')
                            prob = processor.incorporate_detection(this_image, mask,
                                                                   new_segments_info)
                            next_voting_frame += args.detection_every
                            if next_voting_frame >= vid_length:
                                next_voting_frame = vid_length + args.num_voting_frames

                            end.record()
                            torch.cuda.synchronize()
                            total_process_time += (start.elapsed_time(end) / 1000)
                            total_frames += 1

                            if save_this_frame:
                                result_saver.save_mask(
                                    prob,
                                    this_frame_name,
                                    need_resize=need_resize,
                                    shape=shape,
                                    path_to_image=path_to_image,
                                )

                            for frame_info in processor.frame_buffer[1:]:
                                this_image = frame_info.image
                                this_ti = frame_info.ti
                                this_frame_name = frame_info.name
                                save_this_frame = frame_info.save_needed
                                path_to_image = frame_info.path_to_image
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                                prob = processor.step(this_image,
                                                      None,
                                                      None,
                                                      end=(this_ti == vid_length - 1))
                                end.record()
                                torch.cuda.synchronize()
                                total_process_time += (start.elapsed_time(end) / 1000)
                                total_frames += 1

                                if save_this_frame:
                                    result_saver.save_mask(prob,
                                                           this_frame_name,
                                                           need_resize=need_resize,
                                                           shape=shape,
                                                           path_to_image=path_to_image)

                            processor.clear_buffer()
                    else:
                        # standard propagation
                        prob = processor.step(image, None, None, end=(ti == vid_length - 1))
                        end.record()
                        torch.cuda.synchronize()
                        total_process_time += (start.elapsed_time(end) / 1000)
                        total_frames += 1
                        if info['save'][0]:
                            result_saver.save_mask(prob,
                                                   frame,
                                                   need_resize=need_resize,
                                                   shape=shape,
                                                   path_to_image=path_to_image)

                elif temporal_setting == 'online':
                    if ti % args.detection_every == 0:
                        # incorporate new detections
                        assert mask is not None
                        prob = processor.incorporate_detection(image, mask, segments_info)
                    else:
                        # Run the model on this frame
                        prob = processor.step(image, None, None, end=(ti == vid_length - 1))
                    end.record()
                    torch.cuda.synchronize()
                    total_process_time += (start.elapsed_time(end) / 1000)
                    total_frames += 1
                    if info['save'][0]:
                        result_saver.save_mask(prob,
                                               frame,
                                               need_resize=need_resize,
                                               shape=shape,
                                               path_to_image=path_to_image)

                else:
                    raise NotImplementedError

        result_saver.end()
        if is_vipseg:
            # save this for a dataset-level json
            output_json_annotations.append(result_saver.video_json)
        elif is_burst:
            # save this as a video-level json, which we merge later
            with open(path.join(out_path, vid_name, 'pred.json'), 'w') as f:
                json.dump(result_saver.video_json, f)
        elif is_demo:
            # save this as a video-level json in a separate folder
            os.makedirs(path.join(out_path, 'JSONFiles'), exist_ok=True)
            with open(path.join(out_path, 'JSONFiles', f'{vid_name}.json'), 'w') as f:
                json.dump(result_saver.video_json, f, indent=4)

    except Exception as e:
        print(f'Runtime error at {vid_name}')
        print(e)
        raise e  # comment this out if you want

if is_vipseg:
    output_json = {'annotations': output_json_annotations}
    with open(path.join(out_path, 'pred.json'), 'w') as f:
        json.dump(output_json, f)

print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

if is_vipseg:
    vipseg_root = args.vipseg_root
    print('Starting evaluation...')
    merge_stuff(out_path, out_path)

    if not args.no_metrics:
        p1 = Process(target=partial(eval_stq, out_path, f'{vipseg_root}/panomasksRGB',
                                    f'{vipseg_root}/panoptic_gt_VIPSeg_val.json'))
        p1.start()
        eval_vpq(out_path,
                 f'{vipseg_root}/panomasksRGB',
                 f'{vipseg_root}/panoptic_gt_VIPSeg_val.json',
                 num_processes=16)
        p1.join()
elif is_davis:
    if args.postprocess_limit_max_id > 0:
        print('Post-processing DAVIS 2017...')
        limit_max_id(out_path, out_path, max_num_objects=args.postprocess_limit_max_id)
