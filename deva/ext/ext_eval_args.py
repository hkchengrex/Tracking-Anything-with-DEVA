# Evaluation arguments for extensions
from argparse import ArgumentParser


def add_ext_eval_args(parser: ArgumentParser):

    # Grounded Segment Anything
    parser.add_argument('--GROUNDING_DINO_CONFIG_PATH',
                        default='./saves/GroundingDINO_SwinT_OGC.py')

    parser.add_argument('--GROUNDING_DINO_CHECKPOINT_PATH',
                        default='./saves/groundingdino_swint_ogc.pth')

    parser.add_argument('--DINO_THRESHOLD', default=0.35, type=float)
    parser.add_argument('--DINO_NMS_THRESHOLD', default=0.8, type=float)

    # Segment Anything (SAM) models
    parser.add_argument('--SAM_ENCODER_VERSION', default='vit_h')
    parser.add_argument('--SAM_CHECKPOINT_PATH', default='./saves/sam_vit_h_4b8939.pth')

    # Mobile SAM
    parser.add_argument('--MOBILE_SAM_CHECKPOINT_PATH', default='./saves/mobile_sam.pt')

    # Segment Anything (SAM) parameters
    parser.add_argument('--SAM_NUM_POINTS_PER_SIDE',
                        type=int,
                        help='Number of points per side for prompting SAM',
                        default=64)
    parser.add_argument('--SAM_NUM_POINTS_PER_BATCH',
                        type=int,
                        help='Number of points computed per batch',
                        default=64)
    parser.add_argument('--SAM_PRED_IOU_THRESHOLD',
                        type=float,
                        help='(Predicted) IoU threshold for SAM',
                        default=0.88)
    parser.add_argument('--SAM_OVERLAP_THRESHOLD',
                        type=float,
                        help='Overlap threshold for overlapped mask suppression in SAM',
                        default=0.8)


def add_text_default_args(parser):
    parser.add_argument('--img_path', default='./example/vipseg')
    parser.add_argument('--detection_every', type=int, default=5)
    parser.add_argument('--num_voting_frames',
                        default=3,
                        type=int,
                        help='Number of frames selected for voting. only valid in semionline')

    parser.add_argument('--temporal_setting', default='semionline', help='semionline/online')
    parser.add_argument('--max_missed_detection_count', type=int, default=10)
    parser.add_argument('--max_num_objects',
                        default=-1,
                        type=int,
                        help='Max. num of objects to keep in memory. -1 for no limit')
    parser.add_argument('--prompt', type=str, help='Separate classes with a single fullstop')
    parser.add_argument('--sam_variant', default='original', help='mobile/original')
    return parser


def add_auto_default_args(parser):
    parser.add_argument('--img_path', default='./example/vipseg')
    parser.add_argument('--detection_every', type=int, default=5)
    parser.add_argument('--num_voting_frames',
                        default=3,
                        type=int,
                        help='Number of frames selected for voting. only valid in semionline')

    parser.add_argument('--temporal_setting', default='semionline', help='semionline/online')
    parser.add_argument('--max_missed_detection_count', type=int, default=5)
    parser.add_argument('--max_num_objects',
                        default=200,
                        type=int,
                        help='Max. num of objects to keep in memory. -1 for no limit')

    parser.add_argument('--sam_variant', default='original', help='mobile/original')
    parser.add_argument('--suppress_small_objects', action='store_true')

    return parser