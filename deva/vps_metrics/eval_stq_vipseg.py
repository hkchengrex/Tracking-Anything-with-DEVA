# -------------------------------------------------------------------
# Video Panoptic Segmentation
#
# VPQ evaluation code by tube (video segment) matching
# Inference on every frames and evaluation on every 5 frames.
# ------------------------------------------------------------------

# Modified by Rex Cheng Oct 2022 - save results as a .txt file
# Feb 2023 - Ported the update from the official code patch to here
#          - added a functional interface

import argparse
import os
import os.path
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import deva.vps_metrics.segmentation_and_tracking_quality as numpy_stq


def parse_args():
    parser = argparse.ArgumentParser(description='VPSNet eval')
    parser.add_argument('--submit_dir', '-i', type=str, help='test output directory')

    parser.add_argument(
        '--truth_dir',
        type=str,
        default='../VIPSeg/VIPSeg_720P/panomasksRGB',
        help='ground truth directory. Point this to <BASE_DIR>/VIPSeg/VIPSeg_720P/panomasksRGB '
        'after running the conversion script')

    parser.add_argument(
        '--pan_gt_json_file',
        type=str,
        default='../VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json',
        help='ground truth JSON file. Point this to <BASE_DIR>/VIPSeg/VIPSeg_720P/panoptic_gt_'
        'VIPSeg_val.json after running the conversion script')

    args = parser.parse_args()
    return args


# constants
n_classes = 124
ignore_label = 255
bit_shift = 16


def eval_stq(submit_dir, truth_dir, pan_gt_json_file):
    output_dir = submit_dir
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    pan_pred_json_file = os.path.join(submit_dir, 'pred.json')
    with open(pan_pred_json_file, 'r') as f:
        pred_jsons = json.load(f)
    with open(pan_gt_json_file, 'r') as f:
        gt_jsons = json.load(f)

    categories = gt_jsons['categories']

    thing_list_ = []
    for cate_ in categories:
        cat_id = cate_['id']
        isthing = cate_['isthing']
        if isthing:
            thing_list_.append(cat_id)

    stq_metric = numpy_stq.STQuality(n_classes, thing_list_, ignore_label, bit_shift, 2**24)

    pred_annos = pred_jsons['annotations']
    pred_j = {}
    for p_a in pred_annos:
        pred_j[p_a['video_id']] = p_a['annotations']
    gt_annos = gt_jsons['annotations']
    gt_j = {}
    for g_a in gt_annos:
        gt_j[g_a['video_id']] = g_a['annotations']

    pbar = tqdm(gt_jsons['videos'])
    for seq_id, video_images in enumerate(pbar):
        video_id = video_images['video_id']
        pbar.set_description(video_id)

        # print('processing video:{}'.format(video_id))
        gt_image_jsons = video_images['images']
        gt_js = gt_j[video_id]
        pred_js = pred_j[video_id]
        assert len(gt_js) == len(pred_js)

        gt_pans = []
        pred_pans = []
        for imgname_j in gt_image_jsons:
            imgname = imgname_j['file_name']
            image = np.array(Image.open(os.path.join(submit_dir, 'pan_pred', video_id, imgname)))
            pred_pans.append(image)
            image = np.array(Image.open(os.path.join(truth_dir, video_id, imgname)))
            gt_pans.append(image)
        gt_id_to_ins_num_dic = {}
        list_tmp = []
        for segm in gt_js:
            for img_info in segm['segments_info']:
                id_tmp_ = img_info['id']
                if id_tmp_ not in list_tmp:
                    list_tmp.append(id_tmp_)
        for ii, id_tmp_ in enumerate(list_tmp):
            gt_id_to_ins_num_dic[id_tmp_] = ii

        pred_id_to_ins_num_dic = {}
        list_tmp = []
        for segm in pred_js:
            for img_info in segm['segments_info']:
                id_tmp_ = img_info['id']
                if id_tmp_ not in list_tmp:
                    list_tmp.append(id_tmp_)
        for ii, id_tmp_ in enumerate(list_tmp):
            pred_id_to_ins_num_dic[id_tmp_] = ii

        for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(
                list(zip(gt_js, pred_js, gt_pans, pred_pans, gt_image_jsons))):
            #### Step1. Collect frame-level pan_gt, pan_pred, etc.
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256

            ground_truth_instance = np.ones_like(pan_gt) * 255
            ground_truth_semantic = np.ones_like(pan_gt) * 255
            for el in gt_json['segments_info']:
                id_ = el['id']
                cate_id = el['category_id']
                ground_truth_semantic[pan_gt == id_] = cate_id
                ground_truth_instance[pan_gt == id_] = gt_id_to_ins_num_dic[id_]

            ground_truth = ((ground_truth_semantic << bit_shift) + ground_truth_instance)

            prediction_instance = np.ones_like(pan_pred) * 255
            prediction_semantic = np.ones_like(pan_pred) * 255

            for el in pred_json['segments_info']:
                id_ = el['id']
                cate_id = el['category_id']
                prediction_semantic[pan_pred == id_] = cate_id
                prediction_instance[pan_pred == id_] = pred_id_to_ins_num_dic[id_]
            prediction = ((prediction_semantic << bit_shift) + prediction_instance)

            stq_metric.update_state(ground_truth.astype(dtype=np.int32),
                                    prediction.astype(dtype=np.int32), seq_id)
    result = stq_metric.result()
    print('*' * 100)
    print('STQ : {}'.format(result['STQ']))
    print('AQ :{}'.format(result['AQ']))
    print('IoU:{}'.format(result['IoU']))
    print('STQ_per_seq')
    print(result['STQ_per_seq'])
    print('AQ_per_seq')
    print(result['AQ_per_seq'])
    print('ID_per_seq')
    print(result['ID_per_seq'])
    print('Length_per_seq')
    print(result['Length_per_seq'])
    print('*' * 100)

    with open(os.path.join(submit_dir, 'stq.txt'), 'w') as f:
        f.write(f'{result["STQ"]*100:.1f},{result["AQ"]*100:.1f},{result["IoU"]*100:.1f}\n')


if __name__ == "__main__":
    args = parse_args()
    submit_dir = args.submit_dir
    truth_dir = args.truth_dir
    pan_gt_json_file = args.pan_gt_json_file
    eval_stq(submit_dir, truth_dir, pan_gt_json_file)
