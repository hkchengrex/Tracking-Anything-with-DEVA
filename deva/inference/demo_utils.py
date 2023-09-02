import numpy as np
import torch
import torch.nn.functional as F

from deva.dataset.utils import im_normalization
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver


def get_input_frame_for_deva(image_np: np.ndarray, min_side: int) -> torch.Tensor:
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255
    image = im_normalization(image)
    if min_side > 0:
        h, w = image_np.shape[:2]
        scale = min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = image.unsqueeze(0)
        image = F.interpolate(image, (new_h, new_w), mode='bilinear', align_corners=False)[0]
    return image.cuda()


@torch.inference_mode()
def flush_buffer(deva: DEVAInferenceCore, result_saver: ResultSaver) -> None:
    # process all the remaining frames in the buffer
    cfg = deva.config
    new_min_side = cfg['size']
    need_resize = new_min_side > 0

    if 'prompt' in cfg:
        raw_prompt = cfg['prompt']
        prompts = raw_prompt.split('.')
    else:
        prompts = None

    for frame_info in deva.frame_buffer:
        this_image = frame_info.image
        this_frame_name = frame_info.name
        this_image_np = frame_info.image_np
        h, w = this_image_np.shape[:2]
        prob = deva.step(this_image, None, None)
        result_saver.save_mask(prob,
                               this_frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=this_image_np,
                               prompts=prompts)
