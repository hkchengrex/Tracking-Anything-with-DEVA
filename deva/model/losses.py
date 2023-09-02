from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict


def dice_loss(input_mask, cls_gt) -> torch.Tensor:
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:, i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt == (i + 1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.3):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it) -> (torch.Tensor, float):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) /
                                                      (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config['start_warm'], config['end_warm'])

    def compute(self, data, num_objects, it) -> Dict[str, torch.Tensor]:
        losses = defaultdict(int)

        b, t = data['rgb'].shape[:2]

        losses['total_loss'] = 0
        for ti in range(1, t):
            for bi in range(b):
                loss, p = self.bce(data[f'logits_{ti}'][bi:bi + 1, :num_objects[bi] + 1],
                                   data['cls_gt'][bi:bi + 1, ti, 0], it)

                aux_loss = F.cross_entropy(
                    data[f'aux_logits_{ti}'][bi:bi + 1, :num_objects[bi] + 1, 0],
                    data['cls_gt'][bi:bi + 1, ti, 0])

                losses['p'] += p / b / (t - 1)
                losses[f'ce_loss_{ti}'] += loss / b
                losses[f'aux_loss_{ti}'] += aux_loss / b

            losses['total_loss'] += losses['ce_loss_%d' % ti]
            losses['total_loss'] += losses['aux_loss_%d' % ti] * 0.1
            losses[f'dice_loss_{ti}'] = dice_loss(data[f'masks_{ti}'], data['cls_gt'][:, ti, 0])
            losses['total_loss'] += losses[f'dice_loss_{ti}']

        return losses
