# Details on the Demo

Explanation of arguments:
- `DINO_THRESHOLD`: threshold for DINO to be considered as a valid detection
- `SAM_NUM_POINTS_PER_SIDE`: number of points per side to use for automatic grid-based prompting in SAM
- `SAM_NUM_POINTS_PER_BATCH`: number of points prompts to process in parallel in SAM
- `SAM_PRED_IOU_THRESHOLD`: threshold of predicted IoU to be considered as a valid segmentation for SAM
- `SAM_OVERLAP_THRESHOLD`: if suppress_small_objects are enabled, this is the IoU threshold for the suppression. A lower threshold means more segmentation masks (less suppression).
- `amp`: enable mixed precision; is faster and has a lower memory usage
- `chunk_size`: number of objects to be processed in parallel; a higher number means faster inference but higher memory usage
- `size`: internal processing resolution; defaults to 480
- `max_missed_detection_count`: maximum number of consecutive detections that can be missed before an object is deleted from memory
- `max_num_objects`: maximum number of objects that can be tracked at the same time; new objects are ignored if this is exceeded
- `suppress_small_objects`: if enabled, small objects that overlap with large objects are suppressed during the automatic mode; does not matter in the text-prompted mode


Source videos used in our demo:

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/4a00cd0d-f712-447f-82c4-6152addffd6b

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/72e7495c-d5f9-4a8b-b7e8-8714b269e98d

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/337dd073-07eb-4392-9610-c5f6c6b94832

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/e5f6df87-9fd0-4178-8490-00c4b8dc613b

