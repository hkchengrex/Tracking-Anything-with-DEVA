from argparse import ArgumentParser


class Configuration():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
        parser.add_argument('--benchmark', action='store_true')
        # AMP causes problems in training. Not recommended.
        parser.add_argument('--amp', action='store_true')

        # Data parameters
        parser.add_argument('--static_root', help='Static training data root', default='../static')
        parser.add_argument('--bl_root', help='Blender training data root', default='../BL30K')
        parser.add_argument('--yv_root', help='YouTubeVOS data root', default='../YouTube')
        parser.add_argument('--davis_root', help='DAVIS data root', default='../DAVIS')
        parser.add_argument('--ovis_root', help='OVIS data root', default='../OVIS-VOS-train')
        parser.add_argument('--num_workers',
                            help='Total number of dataloader workers across all GPUs processes',
                            type=int,
                            default=16)
        parser.add_argument('--video_data_ratio', default=1.0, type=float)

        parser.add_argument('--pix_feat_dim', default=512, type=int)
        parser.add_argument('--key_dim', default=64, type=int)
        parser.add_argument('--value_dim', default=512, type=int)

        parser.add_argument('--deep_update_prob', default=0.2, type=float)

        # stage 1 and 2 are used previously in MiVOS/STCN/XMem, and not used here
        # we do not train on BL30K for this project, but we keep the naming (s03) for consistency
        parser.add_argument('--stages',
                            help='Training stage (0-static images, 3-DAVIS+YouTubeVOS+OVIS)',
                            default='03')
        parser.add_argument('--clip_grad_norm',
                            help='Clip norm of global gradient at',
                            default=3.0,
                            type=float)
        """
        Stage-specific learning parameters
        Batch sizes are effective -- you don't have to scale them when you scale the number processes
        """
        # Stage 0, static images
        parser.add_argument('--s0_batch_size', default=16, type=int)
        parser.add_argument('--s0_iterations', default=80000, type=int)
        parser.add_argument('--s0_steps', nargs="*", default=[], type=int)
        parser.add_argument('--s0_lr', help='Initial learning rate', default=2e-5, type=float)
        parser.add_argument('--s0_num_ref_frames', default=2, type=int)
        parser.add_argument('--s0_num_frames', default=3, type=int)
        parser.add_argument('--s0_start_warm', default=10000, type=int)
        parser.add_argument('--s0_end_warm', default=35000, type=int)
        parser.add_argument('--s0_schedule', default='constant')

        # Stage 3, DAVIS+YoutubeVOS+OVIS
        parser.add_argument('--s3_batch_size', default=16, type=int)
        parser.add_argument('--s3_iterations', default=150000, type=int)
        parser.add_argument('--s3_steps', nargs="*", default=[120000, 140000], type=int)
        parser.add_argument('--s3_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s3_num_ref_frames', default=3, type=int)
        parser.add_argument('--s3_num_frames', default=8, type=int)
        parser.add_argument('--s3_start_warm', default=10000, type=int)
        parser.add_argument('--s3_end_warm', default=35000, type=int)
        parser.add_argument('--s3_schedule', default='step')

        parser.add_argument('--gamma',
                            help='LR := LR*gamma at every decay step',
                            default=0.1,
                            type=float)
        parser.add_argument('--weight_decay', default=0.001, type=float)

        # Loading
        parser.add_argument('--load_network', help='Path to a pretrained network weights file. ')
        parser.add_argument(
            '--load_checkpoint',
            help='Path to the checkpoint file which contains network weights, optimizer, scheduler, '
            'and the number of iterations. This is used for resuming interrupted training.')

        # Logging information
        parser.add_argument('--log_text_interval', default=100, type=int)
        parser.add_argument('--log_image_interval', default=1500, type=int)
        parser.add_argument('--save_network_interval', default=50000, type=int)
        parser.add_argument('--save_checkpoint_interval', default=50000, type=int)
        parser.add_argument('--exp_id',
                            help='UNIQUE for a training run, set to NULL to disable logging',
                            default='NULL')
        parser.add_argument('--debug',
                            help='Debug mode; logs info at every iteration',
                            action='store_true')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

        # check if the stages are valid
        stage_to_perform = list(self.args['stages'])
        for s in stage_to_perform:
            if s not in ['0', '3']:
                raise NotImplementedError

    def get_stage_parameters(self, stage):
        parameters = {
            'batch_size': self.args['s%s_batch_size' % stage],
            'iterations': self.args['s%s_iterations' % stage],
            'steps': self.args['s%s_steps' % stage],
            'schedule': self.args['s%s_schedule' % stage],
            'lr': self.args['s%s_lr' % stage],
            'num_ref_frames': self.args['s%s_num_ref_frames' % stage],
            'num_frames': self.args['s%s_num_frames' % stage],
            'start_warm': self.args['s%s_start_warm' % stage],
            'end_warm': self.args['s%s_end_warm' % stage],
        }

        return parameters

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)
