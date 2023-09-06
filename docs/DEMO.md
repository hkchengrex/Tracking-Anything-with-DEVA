# Details on the Demo

## Tips on Speeding up Inference

**General Tips:**

- Though innocently looking, reading frames from disk, visualizing the output, and encoding the output as videos can be slow, especially at high resolutions. The script version runs faster than the gradio version because it is more streamlined. Try to disable the visualization with `--no_visualize` (not yet implemented).
- Specifying `--amp` (automatic fixed precision) makes things run faster on most modern GPUs.
- In general, text-prompted inference is faster and more robust than "automatic" inference.
- To speed up the actual processing, we need to speed up either the image model or the propagation model.

**Speed up the image model:**

- The most efficient way is to use the image model less often. This can be achieved by:
  - Using `online` instead of `semionline`, or,
  - Increasing `detection_every`.
- Use a faster image model. For example, Mobile-SAM is faster than SAM. Grounded-Segment-Anything (text-prompt) is faster than automatic SAM. In automatic mode, you can reduce the number of prompting points (`SAM_NUM_POINTS_PER_SIDE`) to reduce the number of queries to SAM.
- In automatic mode, increasing `SAM_NUM_POINTS_PER_BATCH` improves parallelism.

**Speeding up the propagation model:**

- In general, the running time of the propagation model scales linearly with the number of objects (not to be confused with direct proportionality). The best play is thus to reduce the number of objects:
  - Using text-prompt typically generates more relevant objects and fewer overall number of objects.
  - Increasing the thresholds `SAM_PRED_IOU_THRESHOLD` or `DINO_THRESHOLD` reduces the number of detected objects.
  - Reduce `max_missed_detection_count` to delete objects more readily.
  - In automatic mode, enable `suppress_small_objects` to get larger and fewer segments. Note this option has its own overhead.
- Reduce the internal processing resolution `size`. Note this does not affect the image model.
- Try the `fast_memory` option (not yet implemented).
- Increasing `chunk_size` improves parallelism.

## Explanation of arguments

- `DINO_THRESHOLD`: threshold for DINO to be considered as a valid detection
- `SAM_NUM_POINTS_PER_SIDE`: number of points per side to use for automatic grid-based prompting in SAM
- `SAM_NUM_POINTS_PER_BATCH`: number of points prompts to process in parallel in SAM
- `SAM_PRED_IOU_THRESHOLD`: threshold of predicted IoU to be considered as a valid segmentation for SAM
- `SAM_OVERLAP_THRESHOLD`: if suppress_small_objects are enabled, this is the IoU threshold for the suppression. A lower threshold means more segmentation masks (less suppression).
- `detection_every`: number of frames between two consecutive detections; a higher number means faster inference but slower responses to new objects
- `amp`: enable mixed precision; is faster and has a lower memory usage
- `chunk_size`: number of objects to be processed in parallel; a higher number means faster inference but higher memory usage
- `size`: internal processing resolution for the propagation module; defaults to 480
- `max_missed_detection_count`: maximum number of consecutive detections that can be missed before an object is deleted from memory
- `max_num_objects`: maximum number of objects that can be tracked at the same time; new objects are ignored if this is exceeded
- `do_not_pluralize`: pluralization is enabled by default, e.g., turning "pig" into "pigs" and "person" into "people". This seems to increase recall. If you want to disable pluralization, set this to True.
- `suppress_small_objects`: if enabled, small objects that overlap with large objects are suppressed during the automatic mode; does not matter in the text-prompted mode

## Source videos

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/4a00cd0d-f712-447f-82c4-6152addffd6b

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/c556d398-44dd-423b-9ff3-49763eaecd94

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/72e7495c-d5f9-4a8b-b7e8-8714b269e98d

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/337dd073-07eb-4392-9610-c5f6c6b94832

https://github.com/hkchengrex/Tracking-Anything-with-DEVA/assets/7107196/e5f6df87-9fd0-4178-8490-00c4b8dc613b
