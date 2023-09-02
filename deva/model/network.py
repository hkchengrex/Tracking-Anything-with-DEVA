"""
This file defines DEVA, the highest level nn.Module interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

from typing import Dict, Iterable, Union, Tuple
import torch
import torch.nn as nn

from deva.model.modules import *
from deva.model.big_modules import *
from deva.model.memory_utils import *


class DEVA(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.pix_feat_dim = config['pix_feat_dim']
        self.key_dim = config['key_dim']
        self.value_dim = config['value_dim']

        self.pixel_encoder = PixelEncoder(self.pix_feat_dim)
        self.mask_encoder = MaskEncoder(self.pix_feat_dim, self.value_dim)

        # Projection from f16 feature space to key/value space
        self.key_proj = KeyProjection(self.pix_feat_dim, self.key_dim)

        self.mask_decoder = MaskDecoder(self.value_dim)

    def aggregate(self, prob: torch.Tensor, dim: int) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            prob = prob.float()
            new_prob = torch.cat([torch.prod(1 - prob, dim=dim, keepdim=True), prob],
                                 dim).clamp(1e-7, 1 - 1e-7)
            logits = torch.log((new_prob / (1 - new_prob)))

            return logits

    def encode_image(self, image: torch.Tensor) -> (List[torch.Tensor], torch.Tensor):
        multi_scale_features, key_feat = self.pixel_encoder(image)
        return multi_scale_features, key_feat

    def encode_mask(self,
                    image: torch.Tensor,
                    ms_features: Iterable[torch.Tensor],
                    h: torch.Tensor,
                    masks: torch.Tensor,
                    *,
                    is_deep_update: bool = True,
                    chunk_size: int = -1) -> (torch.Tensor, torch.Tensor):
        g16, h16 = self.mask_encoder(image,
                                     ms_features,
                                     h,
                                     masks,
                                     is_deep_update=is_deep_update,
                                     chunk_size=chunk_size)
        return g16, h16

    def transform_key(self,
                      feat: torch.Tensor,
                      *,
                      need_sk: bool = True,
                      need_ek: bool = True) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        key, shrinkage, selection = self.key_proj(feat, need_s=need_sk, need_e=need_ek)
        return key, shrinkage, selection

    # Used in training only.
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key: torch.Tensor, query_selection: torch.Tensor,
                    memory_key: torch.Tensor, memory_shrinkage: torch.Tensor,
                    memory_value: torch.Tensor) -> torch.Tensor:
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        # batch_size, num_objects = memory_id.shape[:2]
        batch_size, num_objects = memory_value.shape[:2]
        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)

        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        # B * (num_objects*CV) * H * W
        memory = readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    def segment(
        self,
        multi_scale_features: Iterable[torch.Tensor],
        memory_readout: torch.Tensor,
        sensory: torch.Tensor,
        last_mask: torch.Tensor,
        *,
        selector: bool = None,
        need_aux: bool = False,
        chunk_size: int = -1,
        update_sensory: bool = True,
        independent_objects: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        multi_scale_features is from the key encoder for skip-connection
        memory_readout is from working/long-term memory
        sensory is the sensory memory
        last_mask is the mask from the last frame, supplementing sensory memory
        selector is 1 if an object exists, and 0 otherwise. We use it to filter padded objects
            during training.
        need_aux is True during training only
        """
        last_mask = F.interpolate(last_mask, size=memory_readout.shape[-2:], mode='area')
        last_mask = last_mask.unsqueeze(2)

        if need_aux:
            sensory, logits, aux_logits = self.mask_decoder(multi_scale_features,
                                                            memory_readout,
                                                            sensory,
                                                            last_mask,
                                                            need_aux=need_aux,
                                                            update_sensory=update_sensory)

            aux_prob = torch.sigmoid(aux_logits)
            if selector is not None:
                aux_prob = aux_prob * selector.unsqueeze(2)

            aux_logits = self.aggregate(aux_prob, dim=1)
            aux_logits = upsample_groups(aux_logits, ratio=16)
            aux_prob = F.softmax(aux_logits, dim=1)
        else:
            sensory, logits = self.mask_decoder(multi_scale_features,
                                                memory_readout,
                                                sensory,
                                                last_mask,
                                                need_aux=need_aux,
                                                chunk_size=chunk_size,
                                                update_sensory=update_sensory)

        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector

        if independent_objects:
            # objects are processed independently
            # individual softmax aggregation with the background
            # should only be used in evaluation mode, assuming batch size is 1
            batch_size, num_objects, h, w = prob.shape
            assert batch_size == 1
            prob = prob.view(num_objects, 1, h, w)
            logits = self.aggregate(prob, dim=1)
            logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
            prob = F.softmax(logits, dim=1)
            # we need to recompute for a single channel of background
            # the actual value of this background does not matter, as long as it does not changes
            # the result of argmax
            background = prob[:, 0].min(dim=0)[0]  # H * W
            prob = prob[:, 1]  # num_objects * H * W
            prob = torch.cat([background.unsqueeze(0), prob], dim=0).unsqueeze(0)
        else:
            # Softmax over all objects
            logits = self.aggregate(prob, dim=1)
            logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
            prob = F.softmax(logits, dim=1)

        if need_aux:
            return sensory, logits, prob, aux_logits, aux_prob
        else:
            return sensory, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_image':
            return self.encode_image(*args, **kwargs)
        elif mode == 'transform_key':
            return self.transform_key(*args, **kwargs)
        elif mode == 'encode_mask':
            return self.encode_mask(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError(mode)

    def load_weights(self, src_dict):
        self.load_state_dict(src_dict)
