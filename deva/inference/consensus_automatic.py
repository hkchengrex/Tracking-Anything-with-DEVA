"""
This file contains the implementation of the consensus where the assoication needs to be inferred.
"""

from typing import List, Literal, Dict
from collections import defaultdict
import torch

from deva.model.memory_utils import *
from deva.model.network import DEVA
from deva.inference.image_feature_store import ImageFeatureStore
from deva.inference.object_info import ObjectInfo
from deva.inference.frame_utils import FrameInfo
from deva.inference.consensus_associated import spatial_alignment
from deva.utils.tensor_utils import pad_divide_by, unpad

import numpy as np

import pulp
try:
    from gurobipy import GRB
    import gurobipy as gp
    use_gurobi = True
except ImportError:
    use_gurobi = False


def solve_with_gurobi(pairwise_iou: np.ndarray, pairwise_iou_indicator: np.ndarray,
                      total_segments: int) -> List[bool]:
    # All experiments in the paper are conducted with gurobi.
    m = gp.Model("solver")
    m.Params.LogToConsole = 0

    # indicator variable
    x = m.addMVar(shape=(total_segments, 1), vtype=GRB.BINARY, name="x")

    # maximize this
    m.setObjective(
        # high support, *2 to compensate because we only computed the upper triangle
        (pairwise_iou @ x).sum() * 2
        # few segments -- the paper says *0.5 but it's later found
        # that the experiments were done with alpha=1 -- should not have a major impact
        - x.sum(),
        GRB.MAXIMIZE)

    # no two selected segments should have >0.5 iou
    m.addConstr((pairwise_iou_indicator * (x @ x.transpose())).sum() == 0, "iou")

    m.optimize()

    results = (x.X > 0.5)[:, 0].tolist()
    return results


def solve_with_pulp(pairwise_iou: np.ndarray, pairwise_iou_indicator: np.ndarray,
                    total_segments: int) -> List[bool]:
    # pulp is a fallback solver; no guarantee that it works the same
    m = pulp.LpProblem('prob', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', range(total_segments), cat=pulp.LpBinary)

    support_objective = pulp.LpAffineExpression([(x[i], pairwise_iou[:, i].sum() * 2)
                                                 for i in range(total_segments)])
    penal_objective = pulp.LpAffineExpression([(x[i], -1) for i in range(total_segments)])
    m += support_objective + penal_objective

    for i in range(total_segments):
        for j in range(i + 1, total_segments):
            if pairwise_iou_indicator[i, j] == 1:
                constraint = pulp.LpConstraint(pulp.LpAffineExpression([(x[i], 1), (x[j], 1)]),
                                               pulp.LpConstraintLE, f'{i}-{j}', 1)
                m += constraint

    # you can change the solver if you have others installed
    m.solve(pulp.PULP_CBC_CMD(msg=0))

    results = [None for _ in range(total_segments)]
    for v in m.variables():
        results[int(v.name[2:])] = v.varValue
    return results


def find_consensus_auto_association(frames: List[FrameInfo],
                                    keyframe_selection: Literal['last', 'middle', 'score',
                                                                'first'] = 'last',
                                    *,
                                    network: DEVA,
                                    store: ImageFeatureStore,
                                    config: Dict) -> (int, torch.Tensor, List[ObjectInfo]):
    global use_gurobi

    time_indices = [f.ti for f in frames]
    images = []
    masks = []
    for f in frames:
        image, pads = pad_divide_by(f.image, 16)
        # masks here have dtype Long and is index-based, i.e., not one-hot
        mask, _ = pad_divide_by(f.mask, 16)
        images.append(image)
        masks.append(mask)

    segments_info = [f.segments_info for f in frames]
    channel_to_id_mappings = []
    internal_id_bookkeeper = 0
    all_new_segments_info = {}
    frame_index_to_seg_info = defaultdict(list)

    # convert all object indices such that indices from different frames do not overlap
    # also convert the masks into one-hot format for propagation
    for i, this_seg_info in enumerate(segments_info):
        new_one_hot_mask = []
        this_channel_mapping = {}
        for si, seg_info in enumerate(this_seg_info):
            old_id = seg_info.id
            internal_id_bookkeeper += 1
            new_id = internal_id_bookkeeper

            # create new object info
            new_seg_info = ObjectInfo(new_id)
            new_seg_info.copy_meta_info(seg_info)
            all_new_segments_info[new_id] = new_seg_info

            # make that into the mask
            new_one_hot_mask.append(masks[i] == old_id)
            this_channel_mapping[si] = new_id
            frame_index_to_seg_info[i].append(new_seg_info)

        if len(new_one_hot_mask) == 0:
            masks[i] = None  # no detected mask
        else:
            masks[i] = torch.stack(new_one_hot_mask, dim=0).float()
        channel_to_id_mappings.append(this_channel_mapping)

    # find a keyframe
    if keyframe_selection == 'last':
        keyframe_i = len(time_indices) - 1
    elif keyframe_selection == 'first':
        keyframe_i = 0
    elif keyframe_selection == 'middle':
        keyframe_i = (len(time_indices) + 1) // 2
    elif keyframe_selection == 'score':
        keyframe_i = None
        raise NotImplementedError
    else:
        raise NotImplementedError

    keyframe_ti = time_indices[keyframe_i]
    keyframe_image = images[keyframe_i]
    keyframe_mask = masks[keyframe_i]

    # project all frames onto the keyframe
    projected_masks = []
    segment_id_to_areas = {}
    segment_id_to_mask = {}
    for ti, image, mask, mapping in zip(time_indices, images, masks, channel_to_id_mappings):
        if mask is None:
            # no detection -> no projection
            projected_masks.append(None)
            continue

        if ti == keyframe_ti:
            # no need to project the keyframe
            projected_mask = torch.cat([torch.ones_like(keyframe_mask[0:1]) * 0.5, keyframe_mask],
                                       dim=0)
        else:
            projected_mask = spatial_alignment(ti, image, mask, keyframe_ti, keyframe_image,
                                               network, store, config)[0]
        projected_mask = unpad(projected_mask, pads)
        # maps the projected mask back into the class index format
        projected_mask = torch.argmax(projected_mask, dim=0)
        remapped_mask = torch.zeros_like(projected_mask)
        for channel_id, object_id in mapping.items():
            # +1 because of background
            this_mask = projected_mask == (channel_id + 1)
            remapped_mask[this_mask] = object_id
            segment_id_to_areas[object_id] = this_mask.sum().item()
            segment_id_to_mask[object_id] = this_mask

        projected_masks.append(remapped_mask.long())

    # compute pairwise iou
    image_area = keyframe_image.shape[-1] * keyframe_image.shape[-2]
    total_segments = internal_id_bookkeeper
    SCALING = 4096
    assert total_segments < SCALING
    # we are filling the upper triangle; diagonal-blocks remain zero
    matching_table = defaultdict(list)
    pairwise_iou = np.zeros((total_segments, total_segments), dtype=np.float32)
    # pairwise_intersection = np.zeros((total_segments, total_segments), dtype=np.float32)
    segments_area = np.zeros((total_segments, 1), dtype=np.float32)
    segments_area[:, 0] = np.array(list(segment_id_to_areas.values()))

    # empty masks in all frames
    if total_segments == 0:
        output_mask = torch.zeros_like(frames[0].mask)
        output_info = []
        return keyframe_ti, output_mask, output_info

    for i in range(len(time_indices)):
        if projected_masks[i] is None:
            continue
        mask1_scaled = projected_masks[i] * SCALING
        for j in range(i + 1, len(time_indices)):
            if projected_masks[j] is None:
                continue
            mask2 = projected_masks[j]
            # vectorized intersection check
            combined = mask1_scaled + mask2

            match_isthing = [None, False, True]  # for isthing
            for isthing_status in match_isthing:
                matched_mask2_id = set()
                for obj1 in frame_index_to_seg_info[i]:
                    mask1_id = obj1.id
                    if obj1.isthing != isthing_status:
                        continue
                    for obj2 in frame_index_to_seg_info[j]:
                        mask2_id = obj2.id
                        # skip if already matched, since we only care IoU>0.5 which is unique
                        if (obj2.isthing != isthing_status) or (mask2_id in matched_mask2_id):
                            continue

                        target_label = mask1_id * SCALING + mask2_id
                        intersection = (combined == target_label).sum().item()
                        if intersection == 0:
                            continue
                        union = segment_id_to_areas[mask1_id] + \
                                segment_id_to_areas[mask2_id] - intersection
                        iou = intersection / union
                        if iou > 0.5:
                            matching_table[mask1_id].append(mask2_id)
                            matching_table[mask2_id].append(mask1_id)
                            matched_mask2_id.add(mask2_id)
                            pairwise_iou[mask1_id - 1, mask2_id - 1] = iou
                            break

    # make symmetric
    pairwise_iou = pairwise_iou + pairwise_iou.T
    # same as >0.5 as we excluded IoU<=0.5
    # 0.49 is used for numerical reasons (probably doesn't actually matter)
    pairwise_iou_indicator = (pairwise_iou > 0.49)
    # suppress low confidence estimation
    pairwise_iou = pairwise_iou * pairwise_iou_indicator
    segments_area /= image_area  # normalization

    if use_gurobi:
        try:
            results = solve_with_gurobi(pairwise_iou, pairwise_iou_indicator, total_segments)
        except gp.GurobiError:
            print('GurobiError, falling back to pulp')
            use_gurobi = False
    if not use_gurobi:
        results = solve_with_pulp(pairwise_iou, pairwise_iou_indicator, total_segments)

    output_mask = torch.zeros_like(frames[0].mask)
    output_info = []
    matched_object_id_to_area = {}
    for channel_id, selected in enumerate(results):
        if selected:
            object_id = channel_id + 1
            matched_object_id_to_area[object_id] = segment_id_to_areas[object_id]

            # merge object info
            new_object_info = all_new_segments_info[object_id]
            for other_object_id in matching_table[object_id]:
                new_object_info.merge(all_new_segments_info[other_object_id])
            output_info.append(new_object_info)

    sorted_by_area = sorted(matched_object_id_to_area.items(), key=lambda x: x[1], reverse=True)
    for object_id, _ in sorted_by_area:
        output_mask[segment_id_to_mask[object_id]] = object_id

    return keyframe_ti, output_mask, output_info
