# DEVA: Tracking Anything with Decoupled Video Segmentation

![titlecard](https://imgur.com/lw15BGH.png)

[Ho Kei Cheng](https://hkchengrex.github.io/), [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh/), [Brian Price](https://www.brianpricephd.com/), [Alexander Schwing](https://www.alexander-schwing.de/), [Joon-Young Lee](https://joonyoung-cv.github.io/)

University of Illinois Urbana-Champaign and Adobe

ICCV 2023

[[arXiV (coming soon)]]() [[PDF]](https://drive.google.com/file/d/1lAgg-j8d6EH1XYUz9htDaZDh4pxuIslb) [[Project Page]](https://hkchengrex.github.io/Tracking-Anything-with-DEVA/)

## Highlights
1. Provide long-term, open-vocabulary video segmentation with text-prompts out-of-the-box.
2. Fairly easy to **integrate your own image model**! Wouldn't you or your reviewers be interested in seeing examples where your image model also works well on videos :smirk:? No finetuning is needed!

## Abstract

We develop a decoupled video segmentation approach (**DEVA**), composed of task-specific image-level segmentation and class/task-agnostic bi-directional temporal propagation.
Due to this design, we only need an image-level model for the target task and a universal temporal propagation model which is trained once and generalizes across tasks.
To effectively combine these two modules, we propose a (semi-)online fusion of segmentation hypotheses from different frames to generate a coherent segmentation.
We show that this decoupled formulation compares favorably to end-to-end approaches in several tasks, most notably in large-vocabulary video panoptic segmentation and open-world video segmentation.

## Demo Videos

### Demo with Grounded Segment Anything (text prompt: "pig"):

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/9a6dbcd1-2c84-45c8-ac0a-4ad31169881f

Source: https://youtu.be/FbK3SL97zf8

### Demo with Grounded Segment Anything (text prompt: "capybara"):

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/2ac5acc2-d160-49be-a013-68ad1d4074c5

Source: https://youtu.be/couz1CrlTdQ

### Demo with Segment Anything (automatic points-in-grid prompting); original video follows DEVA result overlaying the video:

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/ac6ab425-2f49-4438-bcd4-16e4ccfb0d98

Source: DAVIS 2017 validation set "soapbox"

### Demo with Segment Anything on a out-of-domain example; original video follows DEVA result overlaying the video:

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/48542bcd-113c-4454-b512-030df26def08

Source: https://youtu.be/FQQaSyH9hZI

## Installation

(Tested on Ubuntu only)

**Prerequisite:**
- Python 3.7+
- PyTorch 1.12+ and corresponding torchvision

**Clone our repository:**
```bash
git clone https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git
```

**Install with pip:**
```bash
cd Tracking-Anything-with-DEVA
pip install -e .
```

**Download the pretrained models:**
```bash
bash scripts/download_models.sh
```

**(Optional) For fast integer program solving in the semi-online setting:** 

Get your [gurobi](https://www.gurobi.com/) licence which is free for academic use. 
If a license is not found, we fall back to using [PuLP](https://github.com/coin-or/pulp) which is slower and is not rigorously tested by us. All experiments are conducted with gurobi.

**(Optional) For text-prompted/automatic demo:**

Install [our fork of Grounded-Segment-Anything](https://github.com/hkchengrex/Grounded-Segment-Anything). Follow its instructions.

## Quick Start

**With gradio:**
```bash
python demo/demo_gradio.py
```
Then visit the link that popped up on the terminal. If executing on a remote server, try [port forwarding](https://unix.stackexchange.com/questions/115897/whats-ssh-port-forwarding-and-whats-the-difference-between-ssh-local-and-remot).

We have prepared an example in `example/vipseg/12_1mWNahzcsAc` (a clip from the VIPSeg dataset).
The following two scripts segment the example clip using either Grounded Segment Anything with text prompts or SAM with automatic (points in grid) prompting.

(The method runs faster with scripts than with gradio.)

**Script (text-prompted):**
```bash
python demo/demo_with_text.py --chunk_size 4 \
--img_path ./example/vipseg/images/12_1mWNahzcsAc \ 
--amp --temporal_setting semionline \
--size 480 \
--output ./example/output --prompt person.hat.horse
```

**Script (automatic):**
```bash
python demo/demo_automatic.py --chunk_size 4 \
--img_path ./example/vipseg/images/12_1mWNahzcsAc \ 
--amp --temporal_setting semionline \
--size 480 \
--output ./example/output
```

[DEMO.md](docs/DEMO.md) contains more details on the arguments.
You can always look at `deva/inference/eval_args.py` and `deva/ext/ext_eval_args.py` for a full list of arguments.

## Training and Evaluation

1. [Running DEVA with your own detection model.](docs/CUSTOM.md)
2. [Running DEVA with detections to reproduce the benchmark results.](docs/EVALUATION.md)
3. [Training the DEVA model.](docs/TRAINING.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://imgur.com/aouI1WU.png">
  <source media="(prefers-color-scheme: light)" srcset="https://imgur.com/aCbrA9S.png">
  <img alt="separator" src="https://imgur.com/aCbrA9S.png">
</picture>


## Citation

```bibtex
@inproceedings{cheng2023tracking,
  title={Tracking Anything with Decoupled Video Segmentation},
  author={Cheng, Ho Kei and Oh, Seoung Wug and Price, Brian and Schwing, Alexander and Lee, Joon-Young},
  booktitle={ICCV},
  year={2023}
}
```

## References

The demo would not be possible without :heart: from the community:

Grounded Segment Anything: https://github.com/IDEA-Research/Grounded-Segment-Anything

Segment Anything: https://github.com/facebookresearch/segment-anything

XMem: https://github.com/hkchengrex/XMem

Title card generated with OpenPano: https://github.com/ppwwyyxx/OpenPano
