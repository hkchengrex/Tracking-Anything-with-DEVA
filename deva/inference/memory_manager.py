"""
This is conceptually similar to XMem's memory manager, except 
    - Long-term memory for multiple object groups is supported
    - Object deletion is supported
    - Might be a bit less efficient
"""
from typing import Dict, List
import torch

from deva.inference.kv_memory_store import KeyValueMemoryStore
from deva.model.memory_utils import *


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """
    def __init__(self, config: Dict):
        self.sensory_dim = config['value_dim']
        self.top_k = config['top_k']

        self.use_long_term = config['enable_long_term']
        self.count_long_term_usage = config['enable_long_term_count_usage']
        self.chunk_size = config['chunk_size']
        if self.use_long_term:
            self.max_mem_frames = config['max_mid_term_frames']
            self.min_mem_frames = config['min_mid_term_frames']
            self.num_prototypes = config['num_prototypes']
            self.max_long_tokens = config['max_long_term_elements']

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The sensory memory is stored as a dictionary indexed by object ids
        # each of shape C^h x H x W
        self.sensory = {}

        self.work_mem = KeyValueMemoryStore(save_selection=self.use_long_term,
                                            save_usage=self.use_long_term)
        if self.use_long_term:
            self.long_mem = KeyValueMemoryStore(save_usage=self.count_long_term_usage)

        self.config_stale = True
        self.engaged = False

    def update_config(self, config: Dict) -> None:
        self.config_stale = True
        self.sensory_dim = config['value_dim']
        self.top_k = config['top_k']

        assert self.use_long_term == config['enable_long_term'], 'cannot update this'
        assert self.count_long_term_usage == config[
            'enable_long_term_count_usage'], 'cannot update this'

        self.use_long_term = config['enable_long_term']
        self.count_long_term_usage = config['enable_long_term_count_usage']
        if self.use_long_term:
            self.max_mem_frames = config['max_mid_term_frames']
            self.min_mem_frames = config['min_mid_term_frames']
            self.num_prototypes = config['num_prototypes']
            self.max_long_tokens = config['max_long_term_elements']

    def _readout(self, affinity, v) -> torch.Tensor:
        # affinity: N*HW
        # v: C*N or num_objects*C*N
        # returns C*HW or num_objects*C*HW
        if len(v.shape) == 2:
            # single object
            return v @ affinity
        else:
            num_objects, C, N = v.shape
            v = v.view(num_objects * C, N)
            out = v @ affinity
            return out.view(num_objects, C, -1)

    def _long_term_mem_available(self) -> bool:
        return (self.use_long_term and self.long_mem.engaged())

    def _get_sensory_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        return torch.stack([self.sensory[obj] for obj in obj_ids], dim=0)

    def _get_visual_values_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        # All the values that the object ids refer to should have the same shape
        value = torch.stack([self.work_mem.value[obj] for obj in obj_ids], dim=0)
        if self.use_long_term and obj_ids[0] in self.long_mem.value:
            lt_value = torch.stack([self.long_mem.value[obj] for obj in obj_ids], dim=0)
            value = torch.cat([lt_value, value], dim=-1)
        return value

    def match_memory(self, query_key: torch.Tensor,
                     selection: torch.Tensor) -> Dict[int, torch.Tensor]:
        # query_key: 1 x C^k x H x W
        # selection:  1 x C^k x H x W
        # return a dict of memory readouts, indexed by object indices. Each readout is C*H*W
        assert query_key.shape[0] == 1

        h, w = query_key.shape[-2:]
        query_key = query_key[0].flatten(start_dim=1)
        selection = selection[0].flatten(start_dim=1)
        """
        Compute affinity and perform readout
        """
        all_readout_mem = {}
        buckets = self.work_mem.buckets
        for bucket_id, bucket in buckets.items():
            if self.use_long_term and self.long_mem.engaged(bucket_id):
                # Use long-term memory
                long_mem_size = self.long_mem.size(bucket_id)
                memory_key = torch.cat([self.long_mem.key[bucket_id], self.work_mem.key[bucket_id]],
                                       -1)
                shrinkage = torch.cat(
                    [self.long_mem.shrinkage[bucket_id], self.work_mem.shrinkage[bucket_id]], -1)

                similarity = get_similarity(memory_key,
                                            shrinkage,
                                            query_key,
                                            selection,
                                            add_batch_dim=True)
                affinity, usage = do_softmax(similarity,
                                             top_k=self.top_k,
                                             inplace=True,
                                             return_usage=True)
                """
                Record memory usage for working and long-term memory
                """
                # ignore the index return for long-term memory
                work_usage = usage[:, long_mem_size:]
                self.work_mem.update_bucket_usage(bucket_id, work_usage.flatten())

                if self.count_long_term_usage:
                    # ignore the index return for working memory
                    long_usage = usage[:, :long_mem_size]
                    self.long_mem.update_bucket_usage(bucket_id, long_usage.flatten())
            else:
                # no long-term memory
                memory_key = self.work_mem.key[bucket_id]
                shrinkage = self.work_mem.shrinkage[bucket_id]
                similarity = get_similarity(memory_key,
                                            shrinkage,
                                            query_key,
                                            selection,
                                            add_batch_dim=True)

                if self.use_long_term:
                    affinity, usage = do_softmax(similarity,
                                                 top_k=self.top_k,
                                                 inplace=True,
                                                 return_usage=True)
                    self.work_mem.update_bucket_usage(bucket_id, usage.flatten())
                else:
                    affinity = do_softmax(similarity, top_k=self.top_k, inplace=True)

            if self.chunk_size < 1:
                object_chunks = [bucket]
            else:
                object_chunks = [
                    bucket[i:i + self.chunk_size] for i in range(0, len(bucket), self.chunk_size)
                ]

            # apply readout chunk by chunk
            for objects in object_chunks:
                this_msk_value = self._get_visual_values_by_ids(objects)  # num_objects*C*N
                visual_readout = self._readout(affinity,
                                               this_msk_value).view(len(objects), self.CV, h, w)
                for i, obj in enumerate(objects):
                    all_readout_mem[obj] = visual_readout[i]

        return all_readout_mem

    def add_memory(self,
                   key: torch.Tensor,
                   shrinkage: torch.Tensor,
                   value: torch.Tensor,
                   objects: List[int],
                   selection: bool = None) -> None:
        # key: 1*C*H*W
        # value: 1*num_objects*C*H*W
        # objects contains a list of object ids
        self.engaged = True
        if self.H is None or self.config_stale:
            self.config_stale = False
            self.H, self.W = value.shape[-2:]
            self.HW = self.H * self.W
            # convert from num. frames to num. tokens
            if self.use_long_term:
                self.max_work_tokens = self.max_mem_frames * self.HW
                self.min_work_tokens = self.min_mem_frames * self.HW

        # key:   C*N
        # value: num_objects*C*N
        key = key[0].flatten(start_dim=1)
        shrinkage = shrinkage[0].flatten(start_dim=1)
        self.CK = key.shape[0]

        value = value[0].flatten(start_dim=2)
        self.CV = value.shape[1]

        if selection is not None:
            # not really used in non-long-term mode
            selection = selection[0].flatten(start_dim=1)

        # convert tensor into a dict for insertion
        values = {obj: value[obj_id] for obj_id, obj in enumerate(objects)}
        self.work_mem.add(key, values, shrinkage, selection)

        for bucket_id in self.work_mem.buckets.keys():
            # long-term memory cleanup
            if self.use_long_term:
                # Do memory compressed if needed
                if self.work_mem.size(bucket_id) >= self.max_work_tokens:
                    # Remove obsolete features if needed
                    if self.long_mem.size(bucket_id) >= (self.max_long_tokens -
                                                         self.num_prototypes):
                        self.long_mem.remove_obsolete_features(
                            bucket_id, self.max_long_tokens - self.num_prototypes)

                    self.compress_features(bucket_id)

    def purge_except(self, obj_keep_idx: List[int]) -> None:
        # purge certain objects from the memory except the one listed
        self.work_mem.purge_except(obj_keep_idx)
        if self._long_term_mem_available():
            self.long_mem.purge_except(obj_keep_idx)
        self.sensory = {k: v for k, v in self.sensory.items() if k in obj_keep_idx}

        if not self.work_mem.engaged():
            # everything is removed!
            self.engaged = False

    def compress_features(self, bucket_id: int) -> None:
        HW = self.HW

        # perform memory consolidation
        prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
            *self.work_mem.get_all_sliced(bucket_id, HW, -self.min_work_tokens + HW))

        # remove consolidated working memory
        self.work_mem.sieve_by_range(bucket_id,
                                     HW,
                                     -self.min_work_tokens + HW,
                                     min_size=self.min_work_tokens + HW)

        # add to long-term memory
        self.long_mem.add(prototype_key,
                          prototype_value,
                          prototype_shrinkage,
                          selection=None,
                          supposed_bucket_id=bucket_id)

    def consolidation(self, candidate_key: torch.Tensor, candidate_shrinkage: torch.Tensor,
                      candidate_selection: torch.Tensor, candidate_value: Dict[int, torch.Tensor],
                      usage: torch.Tensor) -> (torch.Tensor, Dict[int, torch.Tensor], torch.Tensor):
        # find the indices with max usage
        _, max_usage_indices = torch.topk(usage, k=self.num_prototypes, dim=-1, sorted=True)
        prototype_indices = max_usage_indices.flatten()

        prototype_key = candidate_key[:, prototype_indices]
        prototype_selection = candidate_selection[:, prototype_indices]
        """
        Potentiation step
        """
        similarity = get_similarity(candidate_key,
                                    candidate_shrinkage,
                                    prototype_key,
                                    prototype_selection,
                                    add_batch_dim=True)
        affinity = do_softmax(similarity)

        # readout the values, remove batch dim
        prototype_value = {k: self._readout(affinity, v)[0] for k, v in candidate_value.items()}

        # readout the shrinkage term
        prototype_shrinkage = self._readout(affinity, candidate_shrinkage)[0]  # remove batch dim

        return prototype_key, prototype_value, prototype_shrinkage

    def initialize_sensory_if_needed(self, sample_key: torch.Tensor, ids: List[int]):
        for obj in ids:
            if obj not in self.sensory:
                # also initializes the sensory memory
                h, w = sample_key.shape[-2:]
                self.sensory[obj] = torch.zeros((self.sensory_dim, h, w), device=sample_key.device)

    def update_sensory(self, sensory: torch.Tensor, ids: List[int]):
        # sensory: 1*num_objects*C*H*W
        for obj_id, obj in enumerate(ids):
            self.sensory[obj] = sensory[0, obj_id]

    def get_sensory(self, ids: List[int]):
        # returns 1*num_objects*C*H*W
        return self._get_sensory_by_ids(ids).unsqueeze(0)
