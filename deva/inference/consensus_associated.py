"""
This file contains the implementation of the consensus when the association is already established.
E.g., when we know which mask in frame 1 corresponds to which mask in frame 2.
There is no need to use integer programming for matching.
"""

from typing import List, Dict
import torch

from deva.model.memory_utils import *
from deva.model.network import DEVA
from deva.inference.image_feature_store import ImageFeatureStore
from deva.utils.tensor_utils import pad_divide_by, unpad


def spatial_alignment(src_ti: int, src_image: torch.Tensor, src_mask: torch.Tensor, tar_ti: int,
                      tar_image: torch.Tensor, network: DEVA, store: ImageFeatureStore,
                      config: Dict) -> torch.Tensor:
    """
    src_image/tar_image: 3*H*W
    src_mask: num_objects*H*W

    returns: a segmentation mask of the target image: num_objects*H*W
    """
    num_objects, h, w = src_mask.shape
    src_image = src_image.unsqueeze(0)
    tar_image = tar_image.unsqueeze(0)
    src_mask = src_mask.unsqueeze(0)

    # get source features
    src_ms_features = store.get_ms_features(src_ti, src_image)
    src_key, src_shrinkage, _ = store.get_key(src_ti, src_image)
    # get target features
    tar_ms_features = store.get_ms_features(tar_ti, tar_image)
    tar_key, _, tar_selection = store.get_key(tar_ti, tar_image)

    # encode memory from the source frame
    sensory = torch.zeros((1, num_objects, config['value_dim'], h // 16, w // 16),
                          device=src_key.device)
    value, sensory = network.encode_mask(src_image,
                                         src_ms_features,
                                         sensory,
                                         src_mask,
                                         is_deep_update=True,
                                         chunk_size=config['chunk_size'])

    # key matching
    src_key = src_key.flatten(start_dim=2)
    src_shrinkage = src_shrinkage.flatten(start_dim=2)
    tar_key = tar_key.flatten(start_dim=2)
    tar_selection = tar_selection.flatten(start_dim=2)
    # 1*num_objects*C*H*W -> 1*(num_objects*C)*(H*W)
    value = value.flatten(start_dim=1, end_dim=2).flatten(start_dim=2)

    similarity = get_similarity(src_key, src_shrinkage, tar_key, tar_selection)
    affinity = do_softmax(similarity, top_k=config['top_k'])
    # readout
    memory_readout = value @ affinity
    memory_readout = memory_readout.view(1, num_objects, config['value_dim'], h // 16, w // 16)

    # segmentation
    _, _, tar_mask = network.segment(tar_ms_features,
                                     memory_readout,
                                     sensory,
                                     src_mask,
                                     chunk_size=config['chunk_size'],
                                     update_sensory=False)

    return tar_mask


def _keyframe_objective_from_mask(mask, score, method='high_foreground') -> float:
    # compute a good-to-be-keyframe score of a mask
    if method == 'high_foreground':
        return (mask > 0.8).float().mean()
    elif method == 'score':
        return score
    else:
        raise NotImplementedError


def find_consensus_with_established_association(time_indices: List[int],
                                                images: List[torch.Tensor],
                                                masks: List[torch.Tensor],
                                                network: DEVA,
                                                store: ImageFeatureStore,
                                                config: Dict,
                                                scores: List[float] = None) -> (int, torch.Tensor):

    # apply padding to all images and masks
    for i, (image, mask) in enumerate(zip(images, masks)):
        images[i], pads = pad_divide_by(image, 16)
        masks[i], _ = pad_divide_by(mask, 16)

    # if scores is None, assume uniform (for averaging later on)
    if scores is None:
        scores = [1 for _ in time_indices]
        use_score = False
    else:
        use_score = True
    scores = torch.softmax(torch.Tensor(scores) * 2, dim=0).tolist()

    # first, find a keyframe
    keyframe_objective = float('-inf')
    keyframe_ti = None
    keyframe_image = None
    keyframe_mask = None
    keyframe_score = None

    if use_score:
        # ranking with score
        for ti, image, mask, score in zip(time_indices, images, masks, scores):
            objective = _keyframe_objective_from_mask(mask, score, method='score')
            if objective > keyframe_objective:
                keyframe_objective = objective
                keyframe_ti = ti
                keyframe_image = image
                keyframe_mask = mask
                keyframe_score = score
    else:
        # score-less ranking
        score = None
        for ti, image, mask in zip(time_indices, images, masks):
            objective = _keyframe_objective_from_mask(mask, score, method='high_foreground')
            if objective > keyframe_objective:
                keyframe_objective = objective
                keyframe_ti = ti
                keyframe_image = image
                keyframe_mask = mask
                keyframe_score = score

    if keyframe_score is None:
        keyframe_score = scores[0]

    # then, project all frames onto the keyframe
    # we also project the keyframe onto the keyframe itself for mask refinement
    total_projected_mask = keyframe_mask * keyframe_score
    for ti, image, mask, score in zip(time_indices, images, masks, scores):
        # the keyframe is already added
        if ti == keyframe_ti:
            continue
        projected_mask = spatial_alignment(ti, image, mask, keyframe_ti, keyframe_image, network,
                                           store, config)
        total_projected_mask += projected_mask[0, 1:] * score

    total_projected_mask = unpad(total_projected_mask, pads)
    return keyframe_ti, total_projected_mask
