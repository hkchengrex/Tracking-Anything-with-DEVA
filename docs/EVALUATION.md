# Inference and Evaluation

We provide five evaluation scripts that can be used on common benchmarks. If you are looking to use DEVA on your own data, I suggest you go to [DEMO.md](DEMO.md) instead.

The scripts are:
1. Video Object Segmentation (VOS) evaluation
2. Open-World/Large-Vocabulary/Unsupervised Video Object Segmentation on VIPSeg/BURST/DAVIS 2017
3. Unsupervised Video Object Segmentation (or rather, saliency) on DAVIS 2016
4. Referring Video Object Segmentation (Ref-VOS) evaluation for the Ref-DAVIS dataset
5. Referring Video Object Segmentation (Ref-VOS) evaluation for the Ref-YouTubeVOS dataset

Only (1) is standalone. (2)-(5) require detections from an image model. 

We provide:
1. Pretrained DEVA model (which you can obtain from `scripts/download_models.sh`).
2. Pre-computed detections from image models. [[All can be found here]](https://drive.google.com/drive/folders/1iBJBoKZAFaNYM_6uwBR0Vvc6q0nHXbFR?usp=sharing).
3. Pre-computed outputs from DEVA. [[All can be found here]](https://drive.google.com/drive/folders/1iBJBoKZAFaNYM_6uwBR0Vvc6q0nHXbFR?usp=sharing).
4. Links to the repositories of the image models.

## General Arguments

Here are some of the useful argument options that are shared for all the evaluation scripts.
- Specify `--amp` to use mixed precision for faster processing with a lower memory footprint.
- Specify `--size [xxx]` to change the internal processing resolution. The default is 480.
- Specify `--chunk_size [xxx]` to change the number of objects processed at once. The default is -1, which means all objects are processed in a single pass as a batch.
- Specify `--model [xxx]` to change the path to the pretrained DEVA model. 

## Video Object Segmentation

```bash
python evaluation/eval_vos.py --dataset [dataset] --output [output directory] 
```

- Possible options for [dataset]: `D16` (DAVIS 2016), `D17` (DAVIS 2017), `Y18` (YouTubeVOS-2018), `Y19` (YouTubeVOS-2019), and `G` (Generic dataset, see below).
- Specify `--split test` to test on the DAVIS 2017 test-dev set.
- For generic dataset, additionally specify `--generic_path`. It should point to a directory that contains `JPEGImages` and `Annotations`. In each of those folders, there should be directories of the same name as the video names. Each of those directories should contain the images or annotations for the video. 
- By default, we only use the first-frame annotation in the generic mode. Specify `--use_all_masks` to incorporate new objects (as in the YouTubeVOS dataset).

To get quantitative results:
- DAVIS 2017 validation: [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) or [vos-benchmark](https://github.com/hkchengrex/vos-benchmark).
- DAVIS 2016 validation: [vos-benchmark](https://github.com/hkchengrex/vos-benchmark).
- DAVIS 2017 test-dev: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/6812)
- YouTubeVOS 2018 validation: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/7685)
- YouTubeVOS 2019 validation: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/6066)


*Known issue: We note that DEVA video object segmentation does not perform as well as XMem for very long videos after further testing. This is characterized by a much higher false positive rate when the target object is out-of-view. This might be a consequence of "stable data augmentation" which means the target object is in-view most of the time during training.*

## Open-World/Large-Vocabulary/Unsupervised Video Object Segmentation

**VIPSeg:**

Download VIPSeg from https://github.com/VIPSeg-Dataset/VIPSeg-Dataset and 
convert the data into 720p using their scripts.

Then,
```bash
python evaluation/eval_with_detections.py \
--mask_path [path to detections] --img_path [path to 720p VIPSeg images] \
--dataset vipseg --temporal_setting [online/semionline] \
--output [output directory] --chunk_size 4 
```

Quantitative results should be computed automatically.

Detection models:

- PanoFCN: https://github.com/dvlab-research/PanopticFCN
- Video-K-Net: https://github.com/lxtGH/Video-K-Net
- Mask2Former: https://github.com/facebookresearch/Mask2Former

**BURST:**

Download BURST from https://github.com/Ali2500/BURST-benchmark and subsample every three frames as mentioned in the paper.

Then,
```bash
python evaluation/eval_with_detections.py \
--mask_path [path to detections] --img_path [path to BURST images] \
--dataset burst  --save_all --temporal_setting [online/semionline] \
--output [output directory] --chunk_size 4 
```

Quantitative results can be obtained using https://github.com/Ali2500/BURST-benchmark.

Detection models:
- Mask2Former: https://github.com/facebookresearch/Mask2Former
- EntitySeg: https://github.com/qqlu/Entity

**DAVIS 2017:**

Download DAVIS 2017 from https://davischallenge.org/. 

Then,
```bash
python evaluation/eval_with_detections.py  \
--mask_path [path to detections] --img_path [path to 480p DAVIS images] \
--dataset unsup_davis17 --temporal_setting [online/semionline] \
--output [output directory] --chunk_size 4
```

Quantitative results can be obtained using https://github.com/davisvideochallenge/davis2017-evaluation.

Detection models:
- EntitySeg: https://github.com/qqlu/Entity

**Demo:**

We provide a demo script that runs DEVA on a single video. 
```bash
python evaluation/eval_with_detections.py  \
--mask_path ./example/vipseg/source --img_path ./example/vipseg/images \
--dataset demo --temporal_setting semionline \
--output ./example/output --chunk_size 1
```

## Unsupervised (Salient) Video Object Segmentation

Download DAVIS 2016 from https://davischallenge.org/. 
```bash
python evaluation/eval_saliency.py \
--mask_path [path to detections] --img_path [path to 480p DAVIS images] \
--output [output directory] --imset_path [path to a imset file]
```
The imset file should contain the names of the videos to be evaluated. If you followed our directory structure (in [TRAINING.md](TRAINING.md)), it should be at `../DAVIS/2017/trainval/ImageSets/2016/val.txt`.

Quantitative results can be obtained using https://github.com/davisvideochallenge/davis2017-evaluation.

Detection models:
DIS: https://github.com/xuebinqin/DIS


## Referring Video Object Segmentation

**Referring-DAVIS 2017:**

Download Referring-DAVIS from https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions.

Then, 
```bash
python evaluation/eval_ref_davis.py \
--mask_path [path to detections] --img_path [path to 480p DAVIS images] \
--output [output directory]
```

Note that there are four different expressions for each video. We evaluate each expression separately and report the average.

Quantitative results can be obtained using https://github.com/davisvideochallenge/davis2017-evaluation.

Detection models:
ReferFormer: https://github.com/wjn922/ReferFormer

**Referring-YouTubeVOS:**

Download Referring-YouTubeVOS from https://youtube-vos.org/dataset/rvos/.

Then, 
```bash
python evaluation/eval_ref_youtubevos.py \
--mask_path [path to detections] --img_path [path to YouTubeVOS images] \
--output [output directory]
```

Quantitative results can be obtained from https://competitions.codalab.org/competitions/29139.

Detection models:
ReferFormer: https://github.com/wjn922/ReferFormer

## Understanding Palette (Or, What is Going on with the Colored PNGs?)

See https://github.com/hkchengrex/XMem/blob/main/docs/PALETTE.md.
