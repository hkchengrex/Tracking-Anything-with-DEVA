from typing import List, Optional, Dict, Literal, Iterable
import warnings

import torch
from deva.inference.memory_manager import MemoryManager
from deva.inference.object_manager import ObjectManager
from deva.inference.object_utils import *
from deva.inference.segment_merging import match_and_merge
from deva.inference.image_feature_store import ImageFeatureStore
from deva.inference.frame_utils import FrameInfo
from deva.inference.consensus_automatic import find_consensus_auto_association
from deva.model.network import DEVA

from deva.utils.tensor_utils import pad_divide_by, unpad


class DEVAInferenceCore:
    def __init__(self,
                 network: DEVA,
                 config: Dict,
                 *,
                 image_feature_store: ImageFeatureStore = None):
        self.network = network
        self.mem_every = config['mem_every']
        self.enable_long_term = config['enable_long_term']
        self.chunk_size = config['chunk_size']
        self.max_missed_detection_count = config.get('max_missed_detection_count')
        self.max_num_objects = config.get('max_num_objects')
        self.config = config

        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory = MemoryManager(config=config)
        self.object_manager = ObjectManager()

        if image_feature_store is None:
            self.image_feature_store = ImageFeatureStore(self.network)
        else:
            self.image_feature_store = image_feature_store

        self.last_mask = None

        # for online/semi-online processing
        self.frame_buffer = []

    def enabled_long_id(self) -> None:
        # short id is the default, 1~255, converted to a grayscale mask with a palette
        # long id is usually used by panoptic segmnetation, 255~255**3, converted to a RGB mask
        self.object_manager.use_long_id = True

    @property
    def use_long_id(self):
        return self.object_manager.use_long_id

    def _add_memory(self,
                    image: torch.Tensor,
                    ms_features: Iterable[torch.Tensor],
                    prob: torch.Tensor,
                    key: torch.Tensor,
                    shrinkage: torch.Tensor,
                    selection: torch.Tensor,
                    *,
                    is_deep_update: bool = True) -> None:
        # image: 1*3*H*W
        # ms_features: from the key encoder
        # prob: 1*num_objects*H*W, 0~1
        if prob.shape[1] == 0:
            # nothing to add
            warnings.warn('Empty object mask!', RuntimeWarning)
            return

        self.memory.initialize_sensory_if_needed(key, self.object_manager.all_obj_ids)
        value, sensory = self.network.encode_mask(image,
                                                  ms_features,
                                                  self.memory.get_sensory(
                                                      self.object_manager.all_obj_ids),
                                                  prob,
                                                  is_deep_update=is_deep_update,
                                                  chunk_size=self.chunk_size)
        self.memory.add_memory(key,
                               shrinkage,
                               value,
                               self.object_manager.all_obj_ids,
                               selection=selection)
        self.last_mem_ti = self.curr_ti
        if is_deep_update:
            self.memory.update_sensory(sensory, self.object_manager.all_obj_ids)

    def _segment(self,
                 key: torch.Tensor,
                 selection: torch.Tensor,
                 ms_features: Iterable[torch.Tensor],
                 update_sensory: bool = True) -> torch.Tensor:
        if not self.memory.engaged:
            warnings.warn('Trying to segment without any memory!', RuntimeWarning)
            return torch.zeros((1, key.shape[-2] * 16, key.shape[-1] * 16),
                               device=key.device,
                               dtype=key.dtype)
        memory_readout = self.memory.match_memory(key, selection)
        memory_readout = self.object_manager.realize_dict(memory_readout)
        memory_readout = memory_readout.unsqueeze(0)
        sensory, _, pred_prob_with_bg = self.network.segment(ms_features,
                                                             memory_readout,
                                                             self.memory.get_sensory(
                                                                 self.object_manager.all_obj_ids),
                                                             self.last_mask,
                                                             chunk_size=self.chunk_size,
                                                             update_sensory=update_sensory)
        # remove batch dim
        pred_prob_with_bg = pred_prob_with_bg[0]
        if update_sensory:
            self.memory.update_sensory(sensory, self.object_manager.all_obj_ids)
        return pred_prob_with_bg

    def add_to_temporary_buffer(self, frame_info: FrameInfo) -> None:
        self.frame_buffer.append(frame_info)

    def vote_in_temporary_buffer(
        self,
        keyframe_selection: Literal['last', 'middle', 'score', 'first'] = 'first'
    ) -> (int, torch.Tensor, List[ObjectInfo]):
        projected_ti, projected_mask, projected_info = find_consensus_auto_association(
            self.frame_buffer,
            network=self.network,
            store=self.image_feature_store,
            config=self.config,
            keyframe_selection=keyframe_selection)

        return projected_ti, projected_mask, projected_info

    def clear_buffer(self) -> None:
        # clear buffer
        for f in self.frame_buffer:
            self.image_feature_store.delete(f.ti)
        self.frame_buffer = []

    def incorporate_detection(self,
                              image: torch.Tensor,
                              new_mask: torch.Tensor,
                              segments_info: List[ObjectInfo],
                              *,
                              image_ti_override: bool = None,
                              forward_mask: torch.Tensor = None) -> torch.Tensor:
        # this is used for merging detections from an image-based model
        # it is not used for VOS inference
        self.curr_ti += 1

        if image_ti_override is not None:
            image_ti = image_ti_override
        else:
            image_ti = self.curr_ti

        image, self.pad = pad_divide_by(image, 16)
        new_mask, _ = pad_divide_by(new_mask, 16)
        image = image.unsqueeze(0)  # add the batch dimension

        ms_features = self.image_feature_store.get_ms_features(image_ti, image)
        key, shrinkage, selection = self.image_feature_store.get_key(image_ti, image)

        if forward_mask is None:
            if self.memory.engaged:
                # forward prediction
                prob = self._segment(key, selection, ms_features)
                forward_mask = torch.argmax(prob, dim=0)
            else:
                # initialization
                forward_mask = torch.zeros_like(new_mask)

        # merge masks (Section 3.2.2)
        merged_mask = match_and_merge(forward_mask,
                                      new_mask,
                                      self.object_manager,
                                      segments_info,
                                      max_num_objects=self.max_num_objects,
                                      incremental_mode=(forward_mask is not None))

        # find inactive objects that we need to delete
        purge_activated, tmp_keep_idx, obj_keep_idx = self.object_manager.purge_inactive_objects(
            self.max_missed_detection_count)

        if purge_activated:
            # purge memory
            self.memory.purge_except(obj_keep_idx)
            # purge the merged mask, no background
            new_list = [i - 1 for i in tmp_keep_idx]
            merged_mask = merged_mask[new_list]

        # add mask to memory
        self.last_mask = merged_mask.unsqueeze(0).type_as(key)
        self._add_memory(image, ms_features, self.last_mask, key, shrinkage, selection)
        pred_prob_with_bg = self.network.aggregate(self.last_mask[0], dim=0)

        self.image_feature_store.delete(image_ti)

        return unpad(pred_prob_with_bg, self.pad)

    def step(self,
             image: torch.Tensor,
             mask: torch.Tensor = None,
             objects: Optional[List[int]] = None,
             *,
             hard_mask: bool = True,
             end: bool = False,
             image_ti_override: bool = None,
             delete_buffer: bool = True) -> torch.Tensor:
        """
        image: 3*H*W
        mask: H*W or len(objects)*H*W (if hard) or None
        objects: list of object id, in corresponding order as the mask
                    Ignored if the mask is None.
                    If None, hard_mask must be False.
                        Since we consider each channel in the soft mask to be an object.
        end: if we are at the end of the sequence, we do not need to update memory
            if unsure just set it to False always
        """
        if objects is None and mask is not None:
            assert not hard_mask
            objects = list(range(1, mask.shape[0] + 1))

        self.curr_ti += 1

        if image_ti_override is not None:
            image_ti = image_ti_override
        else:
            image_ti = self.curr_ti

        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension

        is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or
                        (mask is not None)) and (not end)
        # segment when there is no input mask or when the input mask is incomplete
        need_segment = (mask is None) or (not self.object_manager.has_all(objects)
                                          and self.object_manager.num_obj > 0)

        ms_features = self.image_feature_store.get_ms_features(image_ti, image)
        key, shrinkage, selection = self.image_feature_store.get_key(image_ti, image)

        if need_segment:
            pred_prob_with_bg = self._segment(key, selection, ms_features, update_sensory=not end)

        # use the input mask if provided
        if mask is not None:
            # inform the manager of the new objects, get list of temporary id
            corresponding_tmp_ids, _ = self.object_manager.add_new_objects(objects)

            mask, _ = pad_divide_by(mask, 16)
            if need_segment:
                # merge predicted mask with the incomplete input mask
                pred_prob_no_bg = pred_prob_with_bg[1:]
                # use the mutual exclusivity of segmentation
                if hard_mask:
                    pred_prob_no_bg[:, mask > 0] = 0
                else:
                    pred_prob_no_bg[:, mask.max(0) > 0.5] = 0

                new_masks = []
                for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
                    if hard_mask:
                        this_mask = (mask == objects[mask_id]).type_as(pred_prob_no_bg)
                    else:
                        this_mask = mask[tmp_id]
                    if tmp_id >= pred_prob_no_bg.shape[0]:
                        new_masks.append(this_mask.unsqueeze(0))
                    else:
                        # +1 because no background
                        pred_prob_no_bg[tmp_id + 1] = this_mask
                # new_masks are always in the order of tmp_id
                mask = torch.cat([pred_prob_no_bg, *new_masks], dim=0)
            elif hard_mask:
                # simply convert cls to one-hot representation
                mask = torch.stack(
                    [mask == objects[mask_id] for mask_id, _ in enumerate(corresponding_tmp_ids)],
                    dim=0)
            pred_prob_with_bg = self.network.aggregate(mask, dim=0)
            pred_prob_with_bg = torch.softmax(pred_prob_with_bg, dim=0)

        self.last_mask = pred_prob_with_bg[1:].unsqueeze(0)

        # save as memory if needed
        if is_mem_frame:
            self._add_memory(image, ms_features, self.last_mask, key, shrinkage, selection)

        if delete_buffer:
            self.image_feature_store.delete(image_ti)

        return unpad(pred_prob_with_bg, self.pad)
