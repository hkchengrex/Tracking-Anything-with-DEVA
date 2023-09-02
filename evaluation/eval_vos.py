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

try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')

from deva.inference.data.vos_test_datasets import GeneralVOSTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--d16_path', default='../DAVIS/2016')
parser.add_argument('--d17_path', default='../DAVIS/2017')
parser.add_argument('--y18_path', default='../YouTube2018')
parser.add_argument('--y19_path', default='../YouTube')
# For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
parser.add_argument('--generic_path', default='./example/vos')

parser.add_argument('--dataset', help='D16/D17/Y18/Y19/G', default='D17')
parser.add_argument('--split', help='val/test', default='val')
parser.add_argument('--use_all_masks',
                    help='Use all masks in the mask folder for generic evaluation. '
                    'Forced to be True for YouTubeVOS; forced to be False for DAVIS/MOSE.',
                    action='store_true')

# Multi-scale options
parser.add_argument('--save_scores', action='store_true')
parser.add_argument('--flip', action='store_true')

add_common_eval_args(parser)
network, config, args = get_model_and_config(parser)
args.dataset = args.dataset.upper()

if args.output is None:
    args.output = f'../output/{args.dataset}_{args.split}'
    print(f'Output path not provided. Defaulting to {args.output}')
"""
Data preparation
"""
is_youtube = args.dataset.startswith('Y')
is_davis = args.dataset.startswith('D')

if is_youtube or args.save_scores:
    out_path = path.join(args.output, 'Annotations')
else:
    out_path = args.output

if is_youtube:
    if args.dataset == 'Y18':
        yv_path = args.y18_path
    elif args.dataset == 'Y19':
        yv_path = args.y19_path

    if args.split == 'val':
        args.split = 'valid'
        meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='valid', size=args.size)
    elif args.split == 'test':
        meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='test', size=args.size)
    else:
        raise NotImplementedError

elif is_davis:
    if args.dataset == 'D16':
        if args.split == 'val':
            # Set up Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
            meta_dataset = DAVISTestDataset(args.d16_path,
                                            imset='../../2017/trainval/ImageSets/2016/val.txt',
                                            size=args.size)
        else:
            raise NotImplementedError
        palette = None
    elif args.dataset == 'D17':
        if args.split == 'val':
            meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'trainval'),
                                            imset='2017/val.txt',
                                            size=args.size)
        elif args.split == 'test':
            meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'test-dev'),
                                            imset='2017/test-dev.txt',
                                            size=args.size)
        else:
            raise NotImplementedError
elif args.dataset == 'G':
    meta_dataset = GeneralVOSTestDataset(path.join(args.generic_path),
                                         size=args.size,
                                         use_all_masks=args.use_all_masks)

    if not args.save_all:
        args.save_all = True
        print('save_all is forced to be true in generic evaluation mode.')
else:
    raise NotImplementedError

torch.autograd.set_grad_enabled(False)

# Set up loader
meta_loader = meta_dataset.get_datasets()

total_process_time = 0
total_frames = 0

# Start eval
pbar = tqdm(meta_loader, total=len(meta_dataset))
for vid_reader in pbar:

    loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
    vid_name = vid_reader.vid_name
    pbar.set_description(vid_name)
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    config['enable_long_term_count_usage'] = (
        config['enable_long_term']
        and (vid_length / (config['max_mid_term_frames'] - config['min_mid_term_frames']) *
             config['num_prototypes']) >= config['max_long_term_elements'])

    try:
        processor = DEVAInferenceCore(network, config=config)
        first_mask_loaded = False

        for ti, data in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=args.amp):
                image = data['rgb'].cuda()[0]
                mask = data.get('mask')
                if mask is not None:
                    mask = mask.cuda()[0]
                valid_labels = data.get('valid_labels')
                if valid_labels is not None:
                    valid_labels = valid_labels.tolist()[0]
                info = data['info']
                frame = info['frame'][0]
                shape = info['shape']
                need_resize = info['need_resize'][0]

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                # if, for some reason, the first frame is not aligned with the first mask
                if not first_mask_loaded:
                    if mask is not None:
                        first_mask_loaded = True
                    else:
                        # no point to do anything without a mask
                        continue

                if args.flip:
                    image = torch.flip(image, dims=[-1])
                    mask = torch.flip(mask, dims=[-1]) if mask is not None else None

                # Run the model on this frame
                prob = processor.step(image, mask, valid_labels, end=(ti == vid_length - 1))

                # Upsample to original size if needed
                if need_resize:
                    prob = F.interpolate(prob.unsqueeze(1),
                                         shape,
                                         mode='bilinear',
                                         align_corners=False)[:, 0]

                if args.flip:
                    prob = torch.flip(prob, dims=[-1])

                # Probability mask -> index mask
                out_mask = torch.argmax(prob, dim=0)
                out_mask = processor.object_manager.tmp_to_obj_cls(out_mask)

                end.record()
                torch.cuda.synchronize()
                total_process_time += (start.elapsed_time(end) / 1000)
                total_frames += 1

                if args.save_scores:
                    prob = (prob.detach().cpu().numpy() * 255).astype(np.uint8)

                # Save the mask
                if args.save_all or info['save'][0]:
                    this_out_path = path.join(out_path, vid_name)
                    os.makedirs(this_out_path, exist_ok=True)
                    out_img = Image.fromarray(out_mask.cpu().numpy().astype(np.uint8))
                    if vid_reader.get_palette() is not None:
                        out_img.putpalette(vid_reader.get_palette())
                    out_img.save(os.path.join(this_out_path, frame[:-4] + '.png'))

                if args.save_scores:
                    np_path = path.join(args.output, 'Scores', vid_name)
                    os.makedirs(np_path, exist_ok=True)
                    if ti == len(loader) - 1:
                        hkl.dump(processor.object_manager.get_tmp_to_obj_mapping(),
                                 path.join(np_path, f'backward.hkl'),
                                 mode='w')
                    if args.save_all or info['save'][0]:
                        hkl.dump(prob,
                                 path.join(np_path, f'{frame[:-4]}.hkl'),
                                 mode='w',
                                 compression='lzf')

    except Exception as e:
        print(f'Runtime error at {vid_name}')
        print(e)
        raise e

print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

if not args.save_scores:
    if is_youtube:
        print('Making zip for YouTubeVOS...')
        shutil.make_archive(path.join(args.output, path.basename(args.output)), 'zip', args.output,
                            'Annotations')
    elif is_davis and args.split == 'test':
        print('Making zip for DAVIS test-dev...')
        shutil.make_archive(args.output, 'zip', args.output)
