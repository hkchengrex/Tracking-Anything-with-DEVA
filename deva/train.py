import datetime
from os import path
import math

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from deva.model.trainer import Trainer
from deva.dataset.static_dataset import StaticTransformDataset
from deva.dataset.vos_dataset import VOSDataset

from deva.utils.logger import TensorboardLogger
from deva.utils.configuration import Configuration
from deva.utils.load_subset import load_sub_davis, load_sub_yv
"""
Initial set up
"""
# Init distributed environment
distributed.init_process_group(backend="nccl")
print(f'CUDA Device count: {torch.cuda.device_count()}')

# Parse command line arguments
raw_config = Configuration()
raw_config.parse()

if raw_config['benchmark']:
    torch.backends.cudnn.benchmark = True

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f'Rank {local_rank} of {world_size} initialized.')
"""
Auto reloading for multi-stage training in a single script
"""
network_in_memory = None
stages = raw_config['stages']
stages_to_perform = list(stages)
for si, stage in enumerate(stages_to_perform):

    # Set seed to ensure the same initialization
    torch.manual_seed(14159265)
    np.random.seed(14159265)
    random.seed(14159265)

    # Pick stage specific hyperparameters out
    stage_config = raw_config.get_stage_parameters(stage)
    config = dict(**raw_config.args, **stage_config)
    if config['exp_id'] != 'NULL':
        config['exp_id'] = config['exp_id'] + '-s%s' % stages[:si + 1]

    # Batch size and num_workers are both effective,
    # so we get the per-process number by dividing the total with num. processes
    num_gpus = world_size
    if config['batch_size'] // num_gpus * num_gpus != config['batch_size']:
        raise ValueError(
            f'Batch size ({config["batch_size"]}) must be divisible by # GPUs ({num_gpus}).')
    config['batch_size'] //= num_gpus
    config['num_workers'] //= num_gpus
    print(f'We are using {num_gpus} GPUs.')

    print(f'We are now starting to train stage {stage}')
    """
    Model related
    """
    if local_rank == 0:
        # Logging
        if config['exp_id'].lower() != 'null':
            print(f'Logging is enabled for rank {local_rank}')
            long_id = '%s-%s' % (datetime.datetime.now().strftime('%b%d-%H.%M.%S'),
                                 config['exp_id'])
        else:
            long_id = None
        logger = TensorboardLogger(config['exp_id'], long_id)
        logger.log_string('hyperparameters', str(config))

        # Construct the rank 0 model
        model = Trainer(config,
                        logger=logger,
                        save_path=path.join('saves', long_id, config['exp_id'])
                        if long_id is not None else None,
                        local_rank=local_rank,
                        world_size=world_size).train()
    else:
        # Construct model for other ranks
        model = Trainer(config, local_rank=local_rank, world_size=world_size).train()

    # Load pertrained model if needed
    if raw_config['load_checkpoint'] is not None:
        total_iter = model.load_checkpoint(raw_config['load_checkpoint'])
        raw_config['load_checkpoint'] = None
        print('Model checkpoint loaded!')
    else:
        total_iter = 0

    if network_in_memory is not None:
        print('I am loading weights from the previous stage')
        model.load_network_in_memory(network_in_memory)
        network_in_memory = None
    elif raw_config['load_network'] is not None:
        print('I am loading weights from the disk, as listed in configuration')
        model.load_network(raw_config['load_network'])
        raw_config['load_network'] = None
    """
    Dataloader related
    """

    # To re-seed the randomness everytime we start a worker
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 100
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def construct_loader(dataset):
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                        rank=local_rank,
                                                                        shuffle=True)
        train_loader = DataLoader(dataset,
                                  config['batch_size'],
                                  sampler=train_sampler,
                                  num_workers=config['num_workers'],
                                  worker_init_fn=worker_init_fn,
                                  drop_last=True)
        return train_sampler, train_loader

    def renew_vos_loader(max_skip):
        # //5 because we only have annotation for every five frames
        yv_dataset = VOSDataset(
            path.join(yv_root, 'JPEGImages'),
            path.join(yv_root, 'Annotations'),
            max_skip // 5,
            subset=load_sub_yv(),
            num_frames=config['num_frames'],
            data_ratio=config['video_data_ratio'],
        )
        davis_dataset = VOSDataset(
            path.join(davis_root, 'JPEGImages', '480p'),
            path.join(davis_root, 'Annotations', '480p'),
            max_skip,
            subset=load_sub_davis(),
            num_frames=config['num_frames'],
            data_ratio=config['video_data_ratio'],
        )
        ovis_dataset = VOSDataset(
            path.join(ovis_root, 'JPEGImages'),
            path.join(ovis_root, 'Annotations'),
            max_skip // 5,
            subset=None,
            num_frames=config['num_frames'],
            data_ratio=config['video_data_ratio'],
        )
        train_dataset = ConcatDataset([davis_dataset] * 5 + [yv_dataset] + [ovis_dataset] * 3)

        print(f'YouTube dataset size: {len(yv_dataset)}')
        print(f'DAVIS dataset size: {len(davis_dataset)}')
        print(f'OVIS dataset size: {len(ovis_dataset)}')
        print(f'Concat dataset size: {len(train_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(train_dataset)

    """
    Dataset related
    """
    if stage == '0':
        static_root = path.expanduser(config['static_root'])
        # format: path, method (style of storing images), mutliplier
        train_dataset = StaticTransformDataset([
            (path.join(static_root, 'fss'), 0, 1),
            (path.join(static_root, 'DUTS-TR'), 1, 1),
            (path.join(static_root, 'DUTS-TE'), 1, 1),
            (path.join(static_root, 'ecssd'), 1, 1),
            (path.join(static_root, 'BIG_small'), 1, 5),
            (path.join(static_root, 'HRSOD_small'), 1, 5),
        ],
                                               num_frames=config['num_frames'],
                                               max_num_obj=1)
        train_sampler, train_loader = construct_loader(train_dataset)

        print(f'Static dataset size: {len(train_dataset)}')
    else:
        # stage 3, VOS datasets
        max_skip_values = [10, 15, 5, 5]
        increase_skip_fraction = [0.1, 0.3, 0.8, 100]
        yv_root = path.join(path.expanduser(config['yv_root']), 'train')
        davis_root = path.join(path.expanduser(config['davis_root']), '2017', 'trainval')
        ovis_root = path.expanduser(config['ovis_root'])

        train_sampler, train_loader = renew_vos_loader(5)
        renew_loader = renew_vos_loader
    """
    Determine max epoch
    """
    total_epoch = math.ceil(config['iterations'] / len(train_loader))
    current_epoch = total_iter // len(train_loader)
    print(f'We approximately use {total_epoch} epochs.')
    if stage != '0':
        change_skip_iter = [round(config['iterations'] * f) for f in increase_skip_fraction]
        # Skip will only change after an epoch, not in the middle
        print(
            f'The skip value will change approximately at the following iterations: {change_skip_iter[:-1]}'
        )
    """
    Start training
    """
    # Need this to select random bases in different workers
    np.random.seed(np.random.randint(2**30 - 1) + local_rank * 100)
    try:
        while total_iter < config['iterations']:

            # Crucial for randomness!
            train_sampler.set_epoch(current_epoch)
            current_epoch += 1
            print(f'Current epoch: {current_epoch}')

            # Train loop
            model.train()
            for data in train_loader:
                # Update skip if needed
                if stage != '0' and total_iter >= change_skip_iter[0]:
                    while total_iter >= change_skip_iter[0]:
                        cur_skip = max_skip_values[0]
                        max_skip_values = max_skip_values[1:]
                        change_skip_iter = change_skip_iter[1:]
                    print(f'Changing skip to {cur_skip=}')
                    train_sampler, train_loader = renew_loader(cur_skip)
                    break

                if stage != 0 and (config['iterations'] - total_iter <= 5000):
                    model.save_network_interval = 1000

                model.do_pass(data, total_iter)
                total_iter += 1

                if total_iter >= config['iterations']:
                    break
    finally:
        if not config['debug'] and model.logger is not None and total_iter > 5000:
            model.save_network(total_iter)
            model.save_checkpoint(total_iter)

    network_in_memory = model.DEVA.module.state_dict()

torch.cuda.synchronize()
del model
torch.cuda.empty_cache()

distributed.destroy_process_group()
