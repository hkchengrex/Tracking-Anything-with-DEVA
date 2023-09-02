from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import torchvision
import os
from os import path
from PIL import Image, ImagePalette
import pycocotools.mask as mask_util

import numpy as np
import supervision as sv

from deva.utils.pano_utils import ID2RGBConverter
from deva.inference.object_manager import ObjectManager


class ResultSaver:
    def __init__(self,
                 output_root: str,
                 video_name: str,
                 *,
                 dataset: str,
                 object_manager: ObjectManager,
                 palette: Optional[ImagePalette.ImagePalette] = None):
        self.output_root = output_root
        self.video_name = video_name
        self.dataset = dataset.lower()
        self.palette = palette
        self.object_manager = object_manager

        self.need_remapping = False
        self.json_style = None
        self.output_postfix = None
        self.visualize = False

        if self.dataset == 'vipseg':
            self.all_annotations = []
            self.video_json = {'video_id': video_name, 'annotations': self.all_annotations}
            self.need_remapping = True
            self.json_style = 'vipseg'
            self.output_postfix = 'pan_pred'
        elif self.dataset == 'burst':
            self.id2rgb_converter = ID2RGBConverter()
            self.need_remapping = True
            self.all_annotations = []
            dataset_name = path.dirname(video_name)
            seq_name = path.basename(video_name)
            self.video_json = {
                'dataset': dataset_name,
                'seq_name': seq_name,
                'segmentations': self.all_annotations
            }
            self.json_style = 'burst'
        elif self.dataset == 'unsup_davis17':
            self.need_remapping = True
        elif self.dataset == 'ref_davis':
            # nothing special is required
            pass
        elif self.dataset == 'demo':
            self.need_remapping = True
            self.all_annotations = []
            self.video_json = {'annotations': self.all_annotations}
            self.json_style = 'vipseg'
            self.visualize = True
            self.visualize_postfix = 'Visualizations'
            self.output_postfix = 'Annotations'
        elif self.dataset == 'gradio':
            # minimal mode, expect a cv2.VideoWriter to be assigned to self.writer asap
            self.writer = None
            self.need_remapping = True
            self.visualize = True
        else:
            raise NotImplementedError

        if self.need_remapping:
            self.id2rgb_converter = ID2RGBConverter()

    def save_mask(self,
                  prob: torch.Tensor,
                  frame_name: str,
                  need_resize: bool = False,
                  shape: Optional[Tuple[int, int]] = None,
                  save_the_mask: bool = True,
                  image_np: np.ndarray = None,
                  prompts: List[str] = None,
                  path_to_image: str = None):
        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,
                                                                                                 0]

        # Probability mask -> index mask
        mask = torch.argmax(prob, dim=0)

        # remap indices
        if self.need_remapping:
            mask = self.object_manager.tmp_to_obj_cls(mask)

        # record output in the json file
        all_segments_info = self.object_manager.get_current_segments_info()
        if self.json_style == 'vipseg':
            for seg in all_segments_info:
                area = int((mask == seg['id']).sum())
                seg['area'] = area
            # filter out zero-area segments
            all_segments_info = [s for s in all_segments_info if s['area'] > 0]
            # append to video level storage
            this_annotation = {
                'file_name': frame_name[:-4] + '.jpg',
                'segments_info': all_segments_info,
            }
            self.all_annotations.append(this_annotation)
        elif self.json_style == 'burst':
            for seg in all_segments_info:
                seg['mask'] = mask == seg['id']
                seg['area'] = int(seg['mask'].sum())
                coco_mask = mask_util.encode(np.asfortranarray(seg['mask'].cpu().numpy()))
                coco_mask['counts'] = coco_mask['counts'].decode('utf-8')
                seg['rle_mask'] = coco_mask
            # filter out zero-area segments
            all_segments_info = [s for s in all_segments_info if s['area'] > 0]
            # append to video level storage
            this_annotation = {
                'file_name':
                frame_name[:-4] + '.jpg',
                'segmentations': [{
                    'id': seg['id'],
                    'score': seg['score'],
                    'rle': seg['rle_mask'],
                } for seg in all_segments_info],
            }
            self.all_annotations.append(this_annotation)
        elif self.visualize:
            # if we are visualizing, we need to preprocess segment info
            for seg in all_segments_info:
                area = int((mask == seg['id']).sum())
                seg['area'] = area
            # filter out zero-area segments
            all_segments_info = [s for s in all_segments_info if s['area'] > 0]

        # save the mask to disk
        if save_the_mask:
            if self.object_manager.use_long_id:
                out_mask = mask.cpu().numpy().astype(np.uint32)
                rgb_mask = np.zeros((*out_mask.shape[-2:], 3), dtype=np.uint8)
                for id in self.object_manager.all_obj_ids:
                    colored_mask = self.id2rgb_converter._id_to_rgb(id)
                    obj_mask = (out_mask == id)
                    rgb_mask[obj_mask] = colored_mask
                out_img = Image.fromarray(rgb_mask)
            else:
                out_img = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
                if self.palette is not None:
                    out_img.putpalette(self.palette)

            if self.dataset != 'gradio':
                # find a place to save the mask
                if self.output_postfix is not None:
                    this_out_path = path.join(self.output_root, self.output_postfix)
                else:
                    this_out_path = self.output_root
                if self.video_name is not None:
                    this_out_path = path.join(this_out_path, self.video_name)

                os.makedirs(this_out_path, exist_ok=True)
                out_img.save(path.join(this_out_path, frame_name[:-4] + '.png'))

            if self.visualize:
                if image_np is None:
                    if path_to_image is not None:
                        image_np = np.array(Image.open(path_to_image))
                    else:
                        raise ValueError('Cannot visualize without image_np or path_to_image')
                alpha = (out_mask == 0).astype(np.float32) * 0.5 + 0.5
                alpha = alpha[:, :, None]
                blend = (image_np * alpha + rgb_mask * (1 - alpha)).astype(np.uint8)

                if prompts is not None:
                    # draw bounding boxes for the prompts
                    all_masks = []
                    labels = []
                    all_cat_ids = []
                    all_scores = []
                    for seg in all_segments_info:
                        all_masks.append(mask == seg['id'])
                        labels.append(f'{prompts[seg["category_id"]]} {seg["score"]:.2f}')
                        all_cat_ids.append(seg['category_id'])
                        all_scores.append(seg['score'])
                    if len(all_masks) > 0:
                        all_masks = torch.stack(all_masks, dim=0)
                        xyxy = torchvision.ops.masks_to_boxes(all_masks)
                        xyxy = xyxy.cpu().numpy()

                        detections = sv.Detections(xyxy,
                                                   confidence=np.array(all_scores),
                                                   class_id=np.array(all_cat_ids))
                        annotator = sv.BoxAnnotator()
                        blend = annotator.annotate(scene=blend,
                                                   detections=detections,
                                                   labels=labels)

                if self.dataset != 'gradio':
                    # find a place to save the visualization
                    if self.visualize_postfix is not None:
                        this_out_path = path.join(self.output_root, self.visualize_postfix)
                    else:
                        this_out_path = self.output_root
                    if self.video_name is not None:
                        this_out_path = path.join(this_out_path, self.video_name)

                    os.makedirs(this_out_path, exist_ok=True)
                    Image.fromarray(blend).save(path.join(this_out_path, frame_name[:-4] + '.jpg'))
                else:
                    self.writer.write(blend[:, :, ::-1])

        return mask
