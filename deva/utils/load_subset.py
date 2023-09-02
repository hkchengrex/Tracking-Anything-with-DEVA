"""
load_subset.py - Presents a subset of data
DAVIS - only the training set
YouTubeVOS - I manually filtered some erroneous ones out but I haven't checked all
"""


def load_sub_davis(path='deva/utils/davis_subset.txt'):
    with open(path, mode='r') as f:
        subset = set(f.read().splitlines())
    return subset


def load_sub_yv(path='deva/utils/yv_subset.txt'):
    with open(path, mode='r') as f:
        subset = set(f.read().splitlines())
    return subset


def load_referring_yv_val(path='deva/utils/referring-youtubevos-val.txt'):
    with open(path, mode='r') as f:
        subset = set(f.read().splitlines())
    return subset
