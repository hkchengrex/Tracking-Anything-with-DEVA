# Training DEVA

Note that this repository only supports the training of the temporal propagation module. For the image module, please refer to the individual projects.

## Setting Up Data

We put datasets out-of-source, as in XMem. You do not need BL30K. The directory structure should look like this:
```bash
├── Tracking-Anything-with-DEVA
├── DAVIS
│   ├── 2016
│   │   ├── Annotations
│   │   └── ...
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── static
│   ├── BIG_small
│   └── ...
└── YouTube
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   └── valid
└── OVIS-VOS-train
    ├── JPEGImages
    └── Annotations
```

You can try our script `python -m scripts.download_dataset` which might not work 100% of the time due to Google Drive's blocking. If it fails, please download the datasets manually. The links can be found in the script.

## Training Command
The training command is the same as in XMem. We tried training with 4/8 GPUs.
With 8 GPUs, 
```
python -m torch.distributed.run --master_port 25763 --nproc_per_node=8 deva/train.py --exp_id deva_retrain --stage 03
```
- Change `nproc_per_node` to change the number of GPUs. 
- Prepend `CUDA_VISIBLE_DEVICES=...` if you want to use specific GPUs. 
- Change `master_port` if you encounter port collision. 
- `exp_id` is a unique experiment identifier that does not affect how the training is done. 
- Models will be saved in `./saves/`. 
- We simply use the last trained model without model selection.