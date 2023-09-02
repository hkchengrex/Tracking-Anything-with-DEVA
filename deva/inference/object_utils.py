from typing import List, Optional

import torch
from deva.inference.object_info import ObjectInfo
from deva.utils.pano_utils import vipseg_cat_to_isthing


def convert_json_dict_to_objects_info(mask: torch.Tensor,
                                      segments_info: Optional[List],
                                      dataset: str = None) -> List[ObjectInfo]:
    """
    Convert a json dict to a list of object info
    If segments_info is None, we use the unique elements in mask to construct the list
    Otherwise mask is ignored
    """
    if segments_info is not None:
        output = [
            ObjectInfo(
                id=segment['id'],
                category_id=segment.get('category_id'),
                isthing=vipseg_cat_to_isthing[segment.get('category_id')]
                if dataset == 'vipseg' else None,
                score=float(segment['score']) if
                ((dataset == 'burst' or dataset == 'demo') and 'score' in segment) else None)
            for segment in segments_info
        ]
    else:
        # use the mask
        labels = torch.unique(mask)
        labels = labels[labels != 0]
        output = [ObjectInfo(l.item()) for l in labels]

    return output
