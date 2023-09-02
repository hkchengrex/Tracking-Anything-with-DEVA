# Using DEVA with Your Custom Detection Models

There are two main ways to use your own detection models:
1. Online integration. Use a single script that queries the image detection model when needed. This is how the demo works (by querying Grounded Segment Anything or Segment Anything). 
2. Offline integration. Run the image detection model on all frames beforehand and save the results. Then, run DEVA on the saved results. This is how we evaluate DEVA with the benchmark detection models.

From an algorithm point-of-view, both approaches are online/semi-online. There is only an implementation difference.

For (1), look at `demo/demo_automatic` and `demo/demo_with_text`. 

For (2), generate the detections following the data format in `example/vipseg`. There is a json file associated with every segmentation that contains object IDs and meta information. "category_id" and "score" can be optionally included in the json. Then follow the "demo" command listed in [EVALUATION.md](EVALUATION.md). 

You can also "mock" your data as BURST/VIPSeg and run the models as if you are evaluating on BURST/VIPSeg. 
