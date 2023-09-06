from typing import Dict, List, Optional
import torch


class KeyValueMemoryStore:
    """
    Works for key/value pairs type storage
    e.g., working and long-term memory
    """
    def __init__(self, save_selection: bool = False, save_usage: bool = False):
        """
        We store keys and values of objects that first appear in the same frame in a bucket.
        Each bucket contains a set of object ids.
        Each bucket is associated with a single key tensor
            and a dictionary of value tensors indexed by object id.
        """
        self.save_selection = save_selection
        self.save_usage = save_usage

        self.global_bucket_id = 0  # does not reduce even if buckets are removed
        self.buckets: Dict[int, List[int]] = {}  # indexed by bucket id
        self.k: Dict[int, torch.Tensor] = {}  # indexed by bucket id
        self.v: Dict[int, torch.Tensor] = {}  # indexed by object id

        # shrinkage and selection are just like the keys
        self.s = {}
        if self.save_selection:
            self.e = {}

        # usage
        if self.save_usage:
            self.use_cnt = {}  # indexed by bucket id
            self.life_cnt = {}  # indexed by bucket id

    def add(self,
            key: torch.Tensor,
            values: Dict[int, torch.Tensor],
            shrinkage: torch.Tensor,
            selection: torch.Tensor,
            supposed_bucket_id: int = -1) -> None:
        """
        key: C*N
        values: dict of values (C*N), object ids are used as keys
        shrinkage: 1*N
        selection: C*N

        supposed_bucket_id: used to sync the bucket id between working and long-term memory
        if provided, the input should all be in a single bucket indexed by this id
        """
        assert len(key.shape) == 2
        assert len(shrinkage.shape) == 2
        assert not self.save_selection or len(selection.shape) == 2

        # add the value and create new buckets if necessary
        if supposed_bucket_id >= 0:
            enabled_buckets = [supposed_bucket_id]
            bucket_exist = supposed_bucket_id in self.buckets
            for obj, value in values.items():
                if bucket_exist:
                    assert obj in self.v
                    assert obj in self.buckets[supposed_bucket_id]
                    self.v[obj] = torch.cat([self.v[obj], value], -1)
                else:
                    assert obj not in self.v
                    self.v[obj] = value
            self.buckets[supposed_bucket_id] = list(values.keys())
        else:
            new_bucket_id = None
            enabled_buckets = set()
            for obj, value in values.items():
                assert len(value.shape) == 2
                if obj in self.v:
                    self.v[obj] = torch.cat([self.v[obj], value], -1)
                    bucket_used = [
                        bucket_id for bucket_id, object_ids in self.buckets.items()
                        if obj in object_ids
                    ]
                    assert len(bucket_used) == 1  # each object should only be in one bucket
                    enabled_buckets.add(bucket_used[0])
                else:
                    self.v[obj] = value
                    if new_bucket_id is None:
                        # create new bucket
                        new_bucket_id = self.global_bucket_id
                        self.global_bucket_id += 1
                        self.buckets[new_bucket_id] = []
                    # put the new object into the corresponding bucket
                    self.buckets[new_bucket_id].append(obj)
                    enabled_buckets.add(new_bucket_id)

        # create new counters for usage if necessary
        if self.save_usage:
            new_count = torch.zeros((key.shape[1]), device=key.device, dtype=torch.float32)
            new_life = torch.zeros((key.shape[1]), device=key.device, dtype=torch.float32) + 1e-7

        # add the key to every bucket
        for bucket_id in self.buckets:
            if bucket_id not in enabled_buckets:
                # if we are not adding new values to a bucket, we should skip it
                continue
            if bucket_id in self.k:
                self.k[bucket_id] = torch.cat([self.k[bucket_id], key], -1)
                self.s[bucket_id] = torch.cat([self.s[bucket_id], shrinkage], -1)
                if self.save_selection:
                    self.e[bucket_id] = torch.cat([self.e[bucket_id], selection], -1)
                if self.save_usage:
                    self.use_cnt[bucket_id] = torch.cat([self.use_cnt[bucket_id], new_count], -1)
                    self.life_cnt[bucket_id] = torch.cat([self.life_cnt[bucket_id], new_life], -1)
            else:
                self.k[bucket_id] = key
                self.s[bucket_id] = shrinkage
                if self.save_selection:
                    self.e[bucket_id] = selection
                if self.save_usage:
                    self.use_cnt[bucket_id] = new_count
                    self.life_cnt[bucket_id] = new_life

    def update_bucket_usage(self, bucket_id: int, usage: torch.Tensor) -> None:
        # increase all life count by 1
        # increase use of indexed elements
        if not self.save_usage:
            return

        self.use_cnt[bucket_id] += usage.view_as(self.use_cnt[bucket_id])
        self.life_cnt[bucket_id] += 1

    def sieve_by_range(self, bucket_id: int, start: int, end: int, min_size: int) -> None:
        # keep only the elements *outside* of this range (with some boundary conditions)
        # i.e., concat (a[:start], a[end:])
        # bucket with size <= min_size are not modified
        object_ids = self.buckets[bucket_id]

        bucket_num_elements = self.k[bucket_id].shape[-1]
        if bucket_num_elements <= min_size:
            return

        k = self.k[bucket_id]
        s = self.s[bucket_id]
        if self.save_selection:
            e = self.e[bucket_id]
        if self.save_usage:
            use_cnt = self.use_cnt[bucket_id]
            life_cnt = self.life_cnt[bucket_id]

        if end == 0:
            # to accommodate the case where end = -ve 0
            end = bucket_num_elements
        assert end < 0

        self.k[bucket_id] = torch.cat([k[:, :start], k[:, end:]], -1)
        self.s[bucket_id] = torch.cat([s[:, :start], s[:, end:]], -1)
        if self.save_selection:
            self.e[bucket_id] = torch.cat([e[:, :start], e[:, end:]], -1)
        if self.save_usage:
            self.use_cnt[bucket_id] = torch.cat([use_cnt[:start], use_cnt[end:]], -1)
            self.life_cnt[bucket_id] = torch.cat([life_cnt[:start], life_cnt[end:]], -1)
        for obj_id in object_ids:
            v = self.v[obj_id]
            self.v[obj_id] = torch.cat([v[:, :start], v[:, end:]], -1)

    def remove_old_memory(self, bucket_id: int, start_idx: int, max_len: int) -> None:
        self.sieve_by_range(bucket_id, start_idx, -max_len + start_idx, max_len)

    def remove_obsolete_features(self, bucket_id: int, max_size: int) -> None:
        object_ids = self.buckets[bucket_id]

        # normalize with life duration
        usage = self.get_usage(bucket_id).flatten()

        values, _ = torch.topk(usage,
                               k=(self.size(bucket_id) - max_size),
                               largest=False,
                               sorted=True)
        survived = (usage > values[-1])

        self.k[bucket_id] = self.k[bucket_id][:, survived]
        self.s[bucket_id] = self.s[bucket_id][:, survived]
        if self.save_selection:
            # Long-term memory does not store ek so this should not be needed
            self.e[bucket_id] = self.e[bucket_id][:, survived]
        for obj_id in object_ids:
            self.v[obj_id] = self.v[obj_id][:, survived]

        self.use_cnt[bucket_id] = self.use_cnt[bucket_id][survived]
        self.life_cnt[bucket_id] = self.life_cnt[bucket_id][survived]

    def get_usage(self, bucket_id: int) -> torch.Tensor:
        # return normalized usage
        if not self.save_usage:
            raise RuntimeError('I did not count usage!')
        else:
            usage = self.use_cnt[bucket_id] / self.life_cnt[bucket_id]
            return usage

    def get_all_sliced(
        self, bucket_id: int, start: int, end: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], torch.Tensor):
        # return k, sk, ek, value, normalized usage in order, sliced by start and end

        if end == 0:
            # negative 0 would not work as the end index!
            k = self.k[bucket_id][:, start:]
            sk = self.s[bucket_id][:, start:]
            ek = self.e[bucket_id][:, start:] if self.save_selection else None
            value = {obj_id: self.v[obj_id][:, start:] for obj_id in self.buckets[bucket_id]}
            usage = self.get_usage(bucket_id)[start:] if self.save_usage else None
        else:
            k = self.k[bucket_id][:, start:end]
            sk = self.s[bucket_id][:, start:end]
            ek = self.e[bucket_id][:, start:end] if self.save_selection else None
            value = {obj_id: self.v[obj_id][:, start:end] for obj_id in self.buckets[bucket_id]}
            usage = self.get_usage(bucket_id)[start:end] if self.save_usage else None

        return k, sk, ek, value, usage

    def purge_except(self, obj_keep_idx: List[int]):
        # purge certain objects from the memory except the one listed
        obj_keep_idx = set(obj_keep_idx)

        # remove objects that are not in the keep list from the buckets
        buckets_to_remove = []
        for bucket_id, object_ids in self.buckets.items():
            self.buckets[bucket_id] = [obj_id for obj_id in object_ids if obj_id in obj_keep_idx]
            if len(self.buckets[bucket_id]) == 0:
                buckets_to_remove.append(bucket_id)

        # remove object values that are not in the keep list
        self.v = {k: v for k, v in self.v.items() if k in obj_keep_idx}

        # remove buckets that are empty
        for bucket_id in buckets_to_remove:
            del self.buckets[bucket_id]
            del self.k[bucket_id]
            del self.s[bucket_id]
            if self.save_selection:
                del self.e[bucket_id]
            if self.save_usage:
                del self.use_cnt[bucket_id]
                del self.life_cnt[bucket_id]

    def get_v_size(self, obj_id: int) -> int:
        return self.v[obj_id].shape[-1]

    def size(self, bucket_id: int) -> int:
        if bucket_id not in self.k:
            return 0
        else:
            return self.k[bucket_id].shape[-1]

    def engaged(self, bucket_id: Optional[int] = None) -> bool:
        if bucket_id is None:
            return len(self.buckets) > 0
        else:
            return bucket_id in self.buckets

    @property
    def num_objects(self) -> int:
        return len(self.v)

    @property
    def key(self) -> Dict[int, torch.Tensor]:
        return self.k

    @property
    def value(self) -> Dict[int, torch.Tensor]:
        return self.v

    @property
    def shrinkage(self) -> Dict[int, torch.Tensor]:
        return self.s

    @property
    def selection(self) -> Dict[int, torch.Tensor]:
        return self.e

    def __contains__(self, key):
        return key in self.v