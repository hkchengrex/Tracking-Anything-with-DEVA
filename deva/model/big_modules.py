"""
big_modules.py - This file stores higher-level network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

from typing import Iterable, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from deva.model.group_modules import *
from deva.model import resnet
from deva.model.modules import *


class PixelEncoder(nn.Module):
    def __init__(self, pix_feat_dim: int):
        super().__init__()
        self.pix_feat_dim = pix_feat_dim

        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1  # 1/4, 256
        self.layer2 = network.layer2  # 1/8, 512
        self.layer3 = network.layer3  # 1/16, 1024

        # two different projections for key and skip-connection respectively
        self.proj1 = nn.Conv2d(1024, pix_feat_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(1024, pix_feat_dim, kernel_size=1)

    def forward(self, x) -> (Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)  # 1/4, 256
        f8 = self.layer2(f4)  # 1/8, 512
        f16 = self.layer3(f8)  # 1/16, 1024

        return (self.proj1(f16), f8, f4), self.proj2(f16)


class MaskEncoder(nn.Module):
    def __init__(self, pix_feat_dim: int, value_dim: int):
        super().__init__()

        network = resnet.resnet18(pretrained=True, extra_dim=1)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1  # 1/4, 64
        self.layer2 = network.layer2  # 1/8, 128
        self.layer3 = network.layer3  # 1/16, 256

        self.distributor = MainToGroupDistributor()
        self.fuser = GroupFeatureFusionBlock(pix_feat_dim, 256, value_dim, value_dim)

        self.sensory_update = SensoryDeepUpdater(value_dim, value_dim)

    def forward(self,
                image: torch.Tensor,
                ms_features: Iterable[torch.Tensor],
                sensory: torch.Tensor,
                masks: torch.Tensor,
                *,
                is_deep_update: bool = True,
                chunk_size: int = -1) -> (torch.Tensor, torch.Tensor):
        # ms_features are from the key encoder
        # we only use the first one (lowest resolution), following XMem
        g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        if chunk_size < 1 or chunk_size >= num_objects:
            chunk_size = num_objects
            fast_path = True
            new_sensory = sensory
        else:
            if is_deep_update:
                new_sensory = torch.empty_like(sensory)
            else:
                new_sensory = sensory
            fast_path = False

        all_g = []
        for i in range(0, num_objects, chunk_size):
            if fast_path:
                g_chunk = g
            else:
                g_chunk = g[:, i:i + chunk_size]
            actual_chunk_size = g_chunk.shape[1]
            g_chunk = g_chunk.flatten(start_dim=0, end_dim=1)

            g_chunk = self.conv1(g_chunk)
            g_chunk = self.bn1(g_chunk)  # 1/2, 64
            g_chunk = self.maxpool(g_chunk)  # 1/4, 64
            g_chunk = self.relu(g_chunk)

            g_chunk = self.layer1(g_chunk)  # 1/4
            g_chunk = self.layer2(g_chunk)  # 1/8
            g_chunk = self.layer3(g_chunk)  # 1/16

            g_chunk = g_chunk.view(batch_size, actual_chunk_size, *g_chunk.shape[1:])
            g_chunk = self.fuser(ms_features[0], g_chunk)
            all_g.append(g_chunk)
            if is_deep_update:
                if fast_path:
                    new_sensory = self.sensory_update(g_chunk, sensory)
                else:
                    new_sensory[:, i:i + chunk_size] = self.sensory_update(
                        g_chunk, sensory[:, i:i + chunk_size])
        g = torch.cat(all_g, dim=1)

        return g, new_sensory


class MaskDecoder(nn.Module):
    def __init__(self, val_dim: int):
        super().__init__()

        self.fuser = GroupFeatureFusionBlock(512, val_dim, val_dim, val_dim)

        self.sensory_compress = GConv2D(val_dim + 1, val_dim, kernel_size=1)
        self.sensory_update = SensoryUpdater([val_dim, 256, 256 + 1], 512, 512)

        self.decoder_feat_proc = DecoderFeatureProcessor([512, 256], [val_dim, 256])
        self.up_16_8 = MaskUpsampleBlock(val_dim, 256)
        self.up_8_4 = MaskUpsampleBlock(256, 256)

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1)

        self.sensory_linear_pred = LinearPredictor(val_dim, 512)

    def forward(
        self,
        multi_scale_features: Iterable[torch.Tensor],
        memory_readout: torch.Tensor,
        sensory: torch.Tensor,
        last_mask: torch.Tensor,
        *,
        need_aux: bool = False,
        chunk_size: int = -1,
        update_sensory: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        f16, f8, f4 = multi_scale_features
        batch_size, num_objects = memory_readout.shape[:2]

        if need_aux:
            aux_logits = self.sensory_linear_pred(f16, sensory).flatten(start_dim=0, end_dim=1)

        decoder_features = self.decoder_feat_proc([f8, f4])
        if chunk_size < 1 or chunk_size >= num_objects:
            chunk_size = num_objects
            fast_path = True
            new_sensory = sensory
        else:
            if update_sensory:
                new_sensory = torch.empty_like(sensory)
            else:
                new_sensory = sensory
            fast_path = False

        # chunk-by-chunk inference
        all_logits = []
        for i in range(0, num_objects, chunk_size):
            if fast_path:
                p16 = memory_readout + self.sensory_compress(torch.cat([sensory, last_mask], 2))
            else:
                p16 = memory_readout[:, i:i + chunk_size] + self.sensory_compress(
                    torch.cat([sensory[:, i:i + chunk_size], last_mask[:, i:i + chunk_size]], 2))
            actual_chunk_size = p16.shape[1]
            p16 = self.fuser(f16, p16)

            p8 = self.up_16_8(decoder_features[0], p16)
            p4 = self.up_8_4(decoder_features[1], p8)
            with torch.cuda.amp.autocast(enabled=False):
                logits = self.pred(F.relu(p4.flatten(start_dim=0, end_dim=1).float()))

            if update_sensory:
                p4 = torch.cat(
                    [p4, logits.view(batch_size, actual_chunk_size, 1, *logits.shape[-2:])], 2)
                if fast_path:
                    new_sensory = self.sensory_update([p16, p8, p4], sensory)
                else:
                    new_sensory[:,
                                i:i + chunk_size] = self.sensory_update([p16, p8, p4],
                                                                        sensory[:,
                                                                                i:i + chunk_size])
            all_logits.append(logits)

        logits = torch.cat(all_logits, dim=0)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        if need_aux:
            aux_logits = aux_logits.view(batch_size, num_objects, *aux_logits.shape[-3:])

            return new_sensory, logits, aux_logits

        return new_sensory, logits
