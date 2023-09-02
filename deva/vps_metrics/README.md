`eval_stq_vspw.py` and `eval_vpq_vspw.py` are modified from the original evaluation scripts in https://github.com/VIPSeg-Dataset/VIPSeg-Dataset to generate the quantitative results in the GMP paper.

There are a few main modifications:
1. Multiprocessing is implemented to speed up evaluation (by a lot).
2. Masks are loaded on-the-fly to reduce RAM usage.
3. There is no longer a `pan_pred` directory that stores the masks separately. The current script expects a single folder that contains all the video folders with the json file.
4. The print message is modified to be consistent with the notations in the paper and the output text file. `0-frame vpq_stat` becomes `1-frame vpq_stat`;`5-frame vpq_stat` becomes `2-frame vpq_stat`, etc.


I noticed that there has been an update to the origin VIPSeg evaluation script that covers some of these modifications (i.e., point 1 and 2). I developed this version independently with that update and I keep the current version for documentation. They generate the same results in my testing.
