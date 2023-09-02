"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deva.model.network import DEVA
from deva.model.losses import LossComputer
from deva.utils.log_integrator import Integrator
from deva.utils.image_saver import pool_pairs


class Trainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.deep_update_prob = config['deep_update_prob']
        self.local_rank = local_rank

        self.DEVA = nn.parallel.DistributedDataParallel(DEVA(config).cuda(),
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        broadcast_buffers=False)

        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log_string('model_size',
                                   str(sum([param.nelement() for param in self.DEVA.parameters()])))
        self.train_integrator = Integrator(self.logger, distributed=True)
        self.loss_computer = LossComputer(config)

        self.train()
        self.optimizer = optim.AdamW(self.DEVA.parameters(),
                                     lr=config['lr'],
                                     weight_decay=config['weight_decay'])

        if config['schedule'] == 'constant':
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
        elif config['schedule'] == 'poly':
            total_num_iter = config['iterations']
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                         lr_lambda=lambda x:
                                                         (1 - (x / total_num_iter))**0.9)
        elif config['schedule'] == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'],
                                                            config['gamma'])
        else:
            raise NotImplementedError

        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'].float()
        b = frames.shape[0]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        num_objects = first_frame_gt.shape[2]
        selector = data['selector'].unsqueeze(2).unsqueeze(2)

        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            # filler_one = torch.zeros(1, dtype=torch.int64)

            ms_features, feat = self.DEVA('encode_image', frames[:, 0])
            k, s, _ = self.DEVA('transform_key', feat, need_ek=False)

            sensory = torch.zeros((b, num_objects, self.config['value_dim'], *k.shape[-2:]))
            v16, sensory = self.DEVA('encode_mask', frames[:, 0], ms_features, sensory,
                                     first_frame_gt[:, 0])
            masks = first_frame_gt[:, 0]

            # add the time dimension
            keys = k.unsqueeze(2)
            shrinkages = s.unsqueeze(2)
            values = v16.unsqueeze(3)  # B*num_objects*C*T*H*W

            for ti in range(1, self.num_frames):
                ms_features, feat = self.DEVA('encode_image', frames[:, ti])
                k, s, e = self.DEVA('transform_key', feat)
                keys = torch.cat([keys, k.unsqueeze(2)], dim=2)
                shrinkages = torch.cat([shrinkages, s.unsqueeze(2)], dim=2)

                if ti <= self.num_ref_frames:
                    ref_values = values
                    ref_keys = keys[:, :, :ti]
                    ref_shrinkages = shrinkages[:, :, :ti] if shrinkages is not None else None
                else:
                    # pick num_ref_frames random frames
                    # this is not very efficient but I think we would
                    # need broadcasting in gather which we don't have
                    indices = [torch.randperm(ti)[:self.num_ref_frames] for _ in range(b)]
                    ref_values = torch.stack([values[bi, :, :, indices[bi]] for bi in range(b)], 0)
                    ref_keys = torch.stack([keys[bi, :, indices[bi]] for bi in range(b)], 0)
                    ref_shrinkages = torch.stack(
                        [shrinkages[bi, :, indices[bi]]
                         for bi in range(b)], 0) if shrinkages is not None else None

                # Segment frame ti
                memory_readout = self.DEVA('read_memory', k, e if e is not None else None, ref_keys,
                                           ref_shrinkages, ref_values)
                sensory, logits, masks, aux_logits, aux_masks = self.DEVA('segment',
                                                                          ms_features,
                                                                          memory_readout,
                                                                          sensory,
                                                                          masks,
                                                                          selector=selector,
                                                                          need_aux=True)
                # remove background
                masks = masks[:, 1:]
                aux_masks = aux_masks[:, 1:]

                # No need to encode the last frame
                if ti < (self.num_frames - 1):
                    is_deep_update = np.random.rand() < self.deep_update_prob
                    v16, sensory = self.DEVA('encode_mask',
                                             frames[:, ti],
                                             ms_features,
                                             sensory,
                                             masks,
                                             is_deep_update=is_deep_update)
                    values = torch.cat([values, v16.unsqueeze(3)], 3)

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits
                out[f'aux_masks_{ti}'] = aux_masks
                out[f'aux_logits_{ti}'] = aux_logits

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs',
                                                    pool_pairs(images, size, num_filled_objects),
                                                    it)

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time() - self.last_time) /
                                                self.log_text_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.DEVA.parameters(),
                                                       self.config['clip_grad_norm'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.DEVA.parameters(),
                                                       self.config['clip_grad_norm'])
            self.optimizer.step()

        self.scheduler.step()

        if self._do_log:
            if self._is_train:
                self.integrator.add_tensor('grad_norm', grad_norm.item())

    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_{it//1000}K.pth'
        torch.save(self.DEVA.module.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}_checkpoint_{it//1000}K.pth'
        checkpoint = {
            'it': it,
            'network': self.DEVA.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.DEVA.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict):
        self.DEVA.module.load_weights(src_dict)
        print('Network weight loaded from memory.')

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.load_network_in_memory(src_dict)
        print(f'Network weight loaded from {path}')

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.DEVA.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.DEVA.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.DEVA.eval()
        return self
