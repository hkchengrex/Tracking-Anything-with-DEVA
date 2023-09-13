import gradio as gr
import os
from os import path
import tempfile

from argparse import ArgumentParser
import numpy as np
import torch
import cv2
from tqdm import tqdm

from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import flush_buffer
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.automatic_sam import get_sam_model
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args, add_auto_default_args
from deva.ext.automatic_processor import process_frame_automatic as process_frame_auto
from deva.ext.with_text_processor import process_frame_with_text as process_frame_text


def demo_with_text(video: gr.Video, text: str, threshold: float, max_num_objects: int,
                   internal_resolution: int, detection_every: int, max_missed_detection: int,
                   chunk_size: int, sam_variant: str, temporal_setting: str):
    np.random.seed(42)
    torch.autograd.set_grad_enabled(False)
    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    deva_model, cfg, _ = get_model_and_config(parser)
    cfg['prompt'] = text
    cfg['enable_long_term_count_usage'] = True
    cfg['max_num_objects'] = max_num_objects
    cfg['size'] = internal_resolution
    cfg['DINO_THRESHOLD'] = threshold
    cfg['amp'] = True
    cfg['chunk_size'] = chunk_size
    cfg['detection_every'] = detection_every
    cfg['max_missed_detection_count'] = max_missed_detection
    cfg['sam_variant'] = sam_variant
    cfg['temporal_setting'] = temporal_setting
    gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()

    print('Configuration:', cfg)

    # obtain temporary directory
    result_saver = ResultSaver(None, None, dataset='gradio', object_manager=deva.object_manager)
    writer_initizied = False

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ti = 0
    # only an estimate
    with torch.cuda.amp.autocast(enabled=cfg['amp']):
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    if not writer_initizied:
                        h, w = frame.shape[:2]
                        vid_folder = path.join(tempfile.gettempdir(), 'gradio-deva')
                        os.makedirs(vid_folder, exist_ok=True)
                        vid_path = path.join(vid_folder, f'{hash(os.times())}.mp4')
                        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                 (w, h))
                        writer_initizied = True
                        result_saver.writer = writer

                    process_frame_text(deva,
                                       gd_model,
                                       sam_model,
                                       'null.png',
                                       result_saver,
                                       ti,
                                       image_np=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ti += 1
                    pbar.update(1)
                else:
                    break
        flush_buffer(deva, result_saver)
    result_saver.end()
    writer.release()
    cap.release()
    deva.clear_buffer()
    return vid_path


def demo_automatic(video: gr.Video, threshold: float, points_per_side: int, max_num_objects: int,
                   internal_resolution: int, detection_every: int, max_missed_detection: int,
                   sam_num_points: int, chunk_size: int, sam_variant: str, temporal_setting: str,
                   suppress_small_mask: bool):
    np.random.seed(42)
    torch.autograd.set_grad_enabled(False)
    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)
    deva_model, cfg, _ = get_model_and_config(parser)
    cfg['SAM_NUM_POINTS_PER_SIDE'] = int(points_per_side)
    cfg['SAM_NUM_POINTS_PER_BATCH'] = int(sam_num_points)
    cfg['enable_long_term_count_usage'] = True
    cfg['max_num_objects'] = int(max_num_objects)
    cfg['size'] = int(internal_resolution)
    cfg['SAM_PRED_IOU_THRESHOLD'] = threshold
    cfg['amp'] = True
    cfg['chunk_size'] = chunk_size
    cfg['detection_every'] = detection_every
    cfg['max_missed_detection_count'] = max_missed_detection
    cfg['sam_variant'] = sam_variant
    cfg['suppress_small_objects'] = suppress_small_mask
    cfg['temporal_setting'] = temporal_setting
    sam_model = get_sam_model(cfg, 'cuda')

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()

    print('Configuration:', cfg)

    # obtain temporary directory
    result_saver = ResultSaver(None, None, dataset='gradio', object_manager=deva.object_manager)
    writer_initizied = False

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ti = 0
    # only an estimate
    with torch.cuda.amp.autocast(enabled=cfg['amp']):
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    if not writer_initizied:
                        h, w = frame.shape[:2]
                        vid_folder = path.join(tempfile.gettempdir(), 'gradio-deva')
                        os.makedirs(vid_folder, exist_ok=True)
                        vid_path = path.join(vid_folder, f'{hash(os.times())}.mp4')
                        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                 (w, h))
                        writer_initizied = True
                        result_saver.writer = writer

                    process_frame_auto(deva,
                                       sam_model,
                                       'null.png',
                                       result_saver,
                                       ti,
                                       image_np=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ti += 1
                    pbar.update(1)
                else:
                    break
        flush_buffer(deva, result_saver)
    result_saver.end()
    writer.release()
    cap.release()
    deva.clear_buffer()
    return vid_path


text_demo_tab = gr.Interface(
    fn=demo_with_text,
    inputs=[
        gr.Video(),
        gr.Text(label='Prompt (class names delimited by full stops)'),
        gr.Slider(minimum=0.01, maximum=0.99, value=0.35, label='Threshold'),
        gr.Slider(
            minimum=10,
            maximum=1000,
            value=200,
            label='Max num. objects',
            step=1,
        ),
        gr.Slider(
            minimum=384,
            maximum=1080,
            value=480,
            label='Internal resolution',
            step=1,
        ),
        gr.Slider(
            minimum=3,
            maximum=100,
            value=5,
            label='Incorpate detection every [X] frames',
            step=1,
        ),
        gr.Slider(minimum=1,
                  maximum=1000,
                  value=10,
                  step=1,
                  label='Delete segment if undetected for [X] times'),
        gr.Slider(minimum=1,
                  maximum=256,
                  value=8,
                  step=1,
                  label='DEVA number of objects per batch (reduce to save memory)'),
        gr.Dropdown(choices=['mobile', 'original'],
                    label='SAM variant (mobile is faster but less accurate)',
                    value='original'),
        gr.Dropdown(choices=['semionline', 'online'],
                    label='Temporal setting (semionline is slower but less noisy)',
                    value='semionline'),
    ],
    outputs="playable_video",
    examples=[
        [
            'https://user-images.githubusercontent.com/7107196/265518886-e5f6df87-9fd0-4178-8490-00c4b8dc613b.mp4',
            'people.hats.horses',
            0.35,
            200,
            480,
            5,
            5,
            8,
            'original',
            'semionline',
        ],
        [
            'https://user-images.githubusercontent.com/7107196/265518760-72e7495c-d5f9-4a8b-b7e8-8714b269e98d.mp4',
            'people.trees',
            0.35,
            200,
            480,
            5,
            5,
            8,
            'original',
            'semionline',
        ],
        [
            'https://user-images.githubusercontent.com/7107196/265518746-4a00cd0d-f712-447f-82c4-6152addffd6b.mp4',
            'pigs',
            0.35,
            200,
            480,
            5,
            10,
            8,
            'original',
            'semionline',
        ],
        [
            'https://user-images.githubusercontent.com/7107196/265596169-c556d398-44dd-423b-9ff3-49763eaecd94.mp4',
            'capybaras',
            0.35,
            200,
            480,
            5,
            5,
            8,
            'original',
            'semionline',
        ],
    ],
    cache_examples=False,
    title='DEVA: Tracking Anything with Decoupled Video Segmentation (text-prompted)')

auto_demo_tab = gr.Interface(
    fn=demo_automatic,
    inputs=[
        gr.Video(),
        gr.Slider(minimum=0.01, maximum=0.99, value=0.88, label='IoU threshold'),
        gr.Slider(minimum=4, maximum=256, value=64, label='Num. points per side for SAM', step=1),
        gr.Slider(minimum=10,
                  maximum=1000,
                  value=200,
                  label='Max num. objects (reduce to save memory)',
                  step=1),
        gr.Slider(minimum=384, maximum=1080, value=480, label='Internal resolution', step=1),
        gr.Slider(
            minimum=3,
            maximum=100,
            value=5,
            label='Incorpate detection every [X] frames',
            step=1,
        ),
        gr.Slider(minimum=1,
                  maximum=1000,
                  value=5,
                  step=1,
                  label='Delete segment if unseen in [X] detections'),
        gr.Slider(minimum=1,
                  maximum=1024,
                  value=64,
                  step=1,
                  label='SAM number of points per batch (reduce to save memory)'),
        gr.Slider(minimum=1,
                  maximum=256,
                  value=8,
                  step=1,
                  label='DEVA number of objects per batch (reduce to save memory)'),
        gr.Dropdown(choices=['mobile', 'original'],
                    label='SAM variant (mobile is faster but less accurate)',
                    value='original'),
        gr.Dropdown(choices=['semionline', 'online'],
                    label='Temporal setting (semionline is slower but less noisy)',
                    value='semionline'),
        gr.Checkbox(label='Suppress small masks', value=False),
    ],
    outputs="playable_video",
    examples=[
        [
            'https://user-images.githubusercontent.com/7107196/265518760-72e7495c-d5f9-4a8b-b7e8-8714b269e98d.mp4',
            0.88,
            64,
            200,
            480,
            5,
            5,
            64,
            8,
            'original',
            'semionline',
            True,
        ],
        [
            'https://user-images.githubusercontent.com/7107196/265518886-e5f6df87-9fd0-4178-8490-00c4b8dc613b.mp4',
            0.88,
            64,
            200,
            480,
            5,
            5,
            64,
            8,
            'original',
            'semionline',
            False,
        ],
        [
            'https://user-images.githubusercontent.com/7107196/265518805-337dd073-07eb-4392-9610-c5f6c6b94832.mp4',
            0.88,
            64,
            200,
            480,
            5,
            5,
            64,
            8,
            'original',
            'semionline',
            True,
        ]
    ],
    cache_examples=False,
    title='DEVA: Tracking Anything with Decoupled Video Segmentation (automatic)')

if __name__ == "__main__":
    gr.TabbedInterface([text_demo_tab, auto_demo_tab], ["Text prompt", "Automatic"]).launch()
