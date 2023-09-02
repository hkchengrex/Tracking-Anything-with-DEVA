import os
from os import path
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from tqdm import tqdm

from deva.inference.data.saliency_test_datasets import DAVISSaliencyTestDataset
from deva.inference.image_feature_store import ImageFeatureStore
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.consensus_associated import find_consensus_with_established_association
from deva.utils.tensor_utils import pad_divide_by
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--img_path', default='../DAVIS/2016/JPEGImages/480p')
parser.add_argument('--mask_path')
parser.add_argument('--imset_path')
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
meta_dataset = DAVISSaliencyTestDataset(args.img_path, args.mask_path, imset=args.imset_path)
torch.autograd.set_grad_enabled(False)

videos = meta_dataset.get_videos()

total_process_time = 0
total_frames = 0

# Start eval
pbar = tqdm(videos, total=len(meta_dataset))
for vid_name in pbar:
    pbar.set_description(vid_name)

    try:
        """
        initial pass, perform consensus voting and get a keyframe
        """
        image_feature_store = ImageFeatureStore(network)
        vid_reader = meta_dataset.get_offline_sampled_frames(vid_name, config['num_voting_frames'])
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)

        time_indices = []
        images = []
        masks = []
        for ti, data in enumerate(loader):
            time_indices.append(data['info']['time_index'][0].item())
            image = data['rgb'].cuda()[0]
            mask = data['mask'].cuda()[0]
            image, _ = pad_divide_by(image, 16)
            mask, _ = pad_divide_by(mask, 16)
            images.append(image)
            masks.append(mask)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        keyframe_ti, projected_mask = find_consensus_with_established_association(
            time_indices, images, masks, network=network, store=image_feature_store, config=config)
        end.record()
        torch.cuda.synchronize()
        total_process_time += (start.elapsed_time(end) / 1000)
        """
        Backward pass video reader
        """
        backward_vid_reader = meta_dataset.get_partial_video_loader(vid_name,
                                                                    start=-1,
                                                                    end=keyframe_ti + 1,
                                                                    reverse=True)
        """
        Forward pass video reader
        """
        forward_vid_reader = meta_dataset.get_partial_video_loader(vid_name,
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
                config['enable_long_term']
                and (vid_length / (config['max_mid_term_frames'] - config['min_mid_term_frames']) *
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
                                          image_ti_override=image_ti)

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
                    this_out_path = path.join(out_path, vid_name)
                    os.makedirs(this_out_path, exist_ok=True)
                    out_img = Image.fromarray(out_mask.cpu().numpy().astype(np.uint8))
                    out_img.save(os.path.join(this_out_path, frame[:-4] + '.png'))

        with open(path.join(out_path, vid_name, 'key.txt'), 'w') as f:
            f.write(f'options: {time_indices}; keyframe: {keyframe_ti}')

    except Exception as e:
        print(f'Runtime error at {vid_name}')
        print(e)
        raise e

print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')
