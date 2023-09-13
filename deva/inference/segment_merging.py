"""
This file contains the implementation of segment matching and merging (Section 3.2.2).

Match & merge the objects as discussed in the paper 
(Section 3.2.2 Merging Propagation and Consensus)
Also update the object manager
"""

import warnings
from typing import List, Literal, Dict, Optional

import torch
from deva.inference.object_info import ObjectInfo
from deva.inference.object_manager import ObjectManager


def _get_iou(m1, m2, m1_sum, m2_sum) -> (float, float, float):
    intersection = (m1 * m2).sum()
    if intersection < 1e-3:
        return 0, None, None
    union = (m1_sum + m2_sum - intersection)
    return intersection / union, intersection, union


def merge_by_iou(our_masks: Dict[ObjectInfo, torch.Tensor], new_masks: Dict[ObjectInfo,
                                                                            torch.Tensor],
                 our_sums: Dict[ObjectInfo, torch.Tensor], new_sums: Dict[ObjectInfo, torch.Tensor],
                 merged_mask: torch.Tensor, object_manager: ObjectManager,
                 new_segments_info: List[ObjectInfo], isthing_status: Optional[bool],
                 incremental_mode: bool) -> torch.Tensor:
    # meged_mask is edited in-place
    our_to_new_matching = {}
    matched_area = {}
    new_objects = []

    for new_obj in new_segments_info:
        if new_obj.isthing != isthing_status:
            continue
        for our_obj in object_manager.obj_to_tmp_id:
            if (our_obj.isthing != isthing_status) or (our_obj in our_to_new_matching):
                continue
            iou, _, union = _get_iou(new_masks[new_obj], our_masks[our_obj], new_sums[new_obj],
                                     our_sums[our_obj])
            matched = (iou > 0.5)
            if matched:
                our_to_new_matching[our_obj] = new_obj
                matched_area[(our_obj, False)] = union
                break
        else:
            new_objects.append(new_obj)
            matched_area[(new_obj, True)] = new_sums[new_obj]

    # for all unmatched our segment
    for our_obj in object_manager.obj_to_tmp_id:
        if (our_obj.isthing != isthing_status) or (our_obj in our_to_new_matching):
            continue
        matched_area[(our_obj, False)] = our_sums[our_obj]

    # rendering by reversed order of areas
    sorted_by_area = sorted(matched_area.items(), key=lambda x: x[1], reverse=True)
    for (obj, is_new), _ in sorted_by_area:
        if is_new:
            # obj is a new object
            _, corresponding_obj_ids = object_manager.add_new_objects(obj)
            merged_mask[new_masks[obj]] = corresponding_obj_ids[0]
        else:
            # obj is not a new object
            if obj in our_to_new_matching:
                # merge
                new_obj = our_to_new_matching[obj]
                merged_mask[our_masks[obj]] = obj.id
                merged_mask[new_masks[new_obj]] = obj.id
                obj.merge(new_obj)
                obj.unpoke()
            else:
                # copy from our forward mask
                merged_mask[our_masks[obj]] = obj.id
                if incremental_mode:
                    if our_sums[obj] < 1:
                        obj.poke()
                    else:
                        obj.unpoke()
                else:
                    obj.poke()

    return merged_mask


def match_and_merge(our_mask: torch.Tensor,
                    new_mask: torch.Tensor,
                    object_manager: ObjectManager,
                    new_segments_info: List[ObjectInfo],
                    mode: Literal['iou'] = 'iou',
                    max_num_objects: int = -1,
                    incremental_mode: bool = False) -> torch.Tensor:
    """
    our_mask is in temporary ids (consecutive)
    new_mask is in object ids (real ids from json)

    Updates the object manager as a side effect
    mode: 'iou' only
    max_num_objects: maximum number of objects allowed in memory (-1 for no limit)
    incremental_mode: existing masks are not expected to be supported by new masks, 
                        thus we only delete masks when they are not visible for too long, 
                        not when they are unsupported for too long
    """
    mode = mode.lower()

    # separate the masks into one-hot format
    our_mask = our_mask.long()
    new_mask = new_mask.long()
    our_masks = {obj: (our_mask == tmp) for obj, tmp in object_manager.obj_to_tmp_id.items()}
    new_masks = {obj: (new_mask == obj.id) for obj in new_segments_info}

    if max_num_objects > 0 and len(
            object_manager.all_historical_object_ids) + len(new_segments_info) > max_num_objects:
        # too many objects; forcibly deny all new objects
        warnings.warn(
            'Number of objects exceeded maximum (--max_num_objects); discarding new objects')
        new_masks = {}
        new_segments_info = []

    # pre-compute mask sums for IoU computation
    our_sums = {obj: m.sum() for m in our_masks for obj, m in our_masks.items()}
    new_sums = {obj: m.sum() for m in new_masks for obj, m in new_masks.items()}

    # matching
    merged_mask = torch.zeros_like(our_mask)
    match_isthing = [None, False, True]  # for isthing
    # we merge stuff/things/others separately
    for isthing_status in match_isthing:
        if mode == 'iou':
            merged_mask = merge_by_iou(our_masks, new_masks, our_sums, new_sums, merged_mask,
                                       object_manager, new_segments_info, isthing_status,
                                       incremental_mode)
        elif mode == 'engulf':
            raise NotImplementedError('Engulf mode is deprecated')
            merged_mask = merge_by_engulf(our_masks, new_masks, our_sums, new_sums, merged_mask,
                                          object_manager, new_segments_info, isthing_status,
                                          engulf_threshold)

    merged_mask = object_manager.make_one_hot(merged_mask)
    return merged_mask
