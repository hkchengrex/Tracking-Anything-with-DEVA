import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from tqdm import tqdm

from deva.inference.data.referring_test_datasets import ReferringYouTubeVOSTestDataset
from deva.inference.image_feature_store import ImageFeatureStore
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.consensus_associated import find_consensus_with_established_association
from deva.utils.load_subset import load_referring_yv_val
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--img_path', default='../YouTube/all_frames/valid_all_frames/JPEGImages')
parser.add_argument('--mask_path')
parser.add_argument('--json_path',
                    default='../YouTube/meta_expressions/valid/meta_expressions.json')
parser.add_argument('--num_voting_frames',
                    default=10,
                    type=int,
                    help='Number of frames selected for the initial consensus voting')
add_common_eval_args(parser)
network, config, args = get_model_and_config(parser)
"""
Data preparation
"""
out_path = args.output
meta_dataset = ReferringYouTubeVOSTestDataset(args.img_path, args.mask_path, args.json_path)
torch.autograd.set_grad_enabled(False)

videos = meta_dataset.get_videos()
video_subset = load_referring_yv_val()
print(f'Subset size: {len(video_subset)}')

total_process_time = 0
total_frames = 0

# Start eval
pbar = tqdm(video_subset)
for vid_name in pbar:
    pbar.set_description(vid_name)
    objects = meta_dataset.get_objects(vid_name)
    video_scores = meta_dataset.get_scores(vid_name)
    image_feature_store = ImageFeatureStore(network, no_warning=True)
    for object_name in objects:
        try:
            """
            initial pass, perform consensus voting and get a keyframe
            """
            object_scores = video_scores[object_name]
            vid_reader = meta_dataset.get_offline_sampled_frames(vid_name, object_name,
                                                                 config['num_voting_frames'])
            loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)

            time_indices = []
            images = []
            masks = []
            scores = []
            for ti, data in enumerate(loader):
                image_ti = data['info']['time_index'][0].item()
                time_indices.append(image_ti)
                image = data['rgb'].cuda()[0]
                mask = data['mask'].cuda()[0]
                images.append(image)
                masks.append(mask)

                frame_name = data['info']['frame'][0][:-4]
                scores.append(object_scores[frame_name])

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            keyframe_ti, projected_mask = find_consensus_with_established_association(
                time_indices,
                images,
                masks,
                scores=scores,
                network=network,
                store=image_feature_store,
                config=config)
            end.record()
            torch.cuda.synchronize()
            total_process_time += (start.elapsed_time(end) / 1000)
            """
            Backward pass video reader
            """
            backward_vid_reader = meta_dataset.get_partial_video_loader(vid_name,
                                                                        object_name,
                                                                        start=-1,
                                                                        end=keyframe_ti + 1,
                                                                        reverse=True)
            """
            Forward pass video reader
            """
            forward_vid_reader = meta_dataset.get_partial_video_loader(vid_name,
                                                                       object_name,
                                                                       start=keyframe_ti,
                                                                       end=-1,
                                                                       reverse=False)
            """
            Running them in combination
            """
            vid_readers = [backward_vid_reader, forward_vid_reader]
            for vid_reader in vid_readers:

                loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
                vid_length = len(loader)
                # no need to count usage for LT if the video is not that long anyway
                config['enable_long_term_count_usage'] = (
                    config['enable_long_term'] and
                    (vid_length / (config['max_mid_term_frames'] - config['min_mid_term_frames']) *
                     config['num_prototypes']) >= config['max_long_term_elements'])

                processor = DEVAInferenceCore(network,
                                              config=config,
                                              image_feature_store=image_feature_store)

                for ti, data in enumerate(loader):
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        image = data['rgb'].cuda()[0]
                        info = data['info']
                        frame = info['frame'][0]
                        shape = info['shape']
                        need_resize = info['need_resize'][0]
                        image_ti = info['time_index'][0].item()

                        if image_ti == keyframe_ti:
                            mask = projected_mask
                        else:
                            mask = None

                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()

                        # Run the model on this frame
                        prob = processor.step(image,
                                              mask,
                                              end=(ti == vid_length - 1),
                                              hard_mask=False,
                                              image_ti_override=image_ti,
                                              delete_buffer=False)

                        # Upsample to original size if needed
                        if need_resize:
                            prob = F.interpolate(prob.unsqueeze(1),
                                                 shape,
                                                 mode='bilinear',
                                                 align_corners=False)[:, 0]

                        out_mask = (prob[1] > prob[0]).float() * 255

                        end.record()
                        torch.cuda.synchronize()
                        total_process_time += (start.elapsed_time(end) / 1000)
                        total_frames += 1

                        # Save the mask
                        if args.save_all or info['save'][0]:
                            this_out_path = path.join(out_path, 'Annotations', vid_name,
                                                      object_name)
                            os.makedirs(this_out_path, exist_ok=True)
                            out_img = Image.fromarray(out_mask.cpu().numpy().astype(np.uint8))
                            out_img.save(os.path.join(this_out_path, frame[:-4] + '.png'))

            with open(path.join(out_path, 'Annotations', vid_name, object_name, 'key.txt'),
                      'w') as f:
                f.write(f'options: {time_indices}; keyframe: {keyframe_ti}')

        except Exception as e:
            print(f'Runtime error at {vid_name}')
            print(e)
            raise e

print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

print('Making zip for YouTubeVOS...')
shutil.make_archive(path.join(args.output, path.basename(args.output)), 'zip', args.output,
                    'Annotations')
