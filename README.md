# Metrics evaluation
## Set up environment
```
conda create -n bpm python=3.8
pip install -r requirements.txt
```
Then please follow [Grounded_Segment_Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main?tab=readme-ov-file#running_man-grounded-sam-detect-and-segment-everything-with-text-prompt) to download pretrained weights


## Run metrics
Run BPM metric
```
./run_metrics.sh
```
Run other metrics (except for gpt)
```
cd filter_script
./run_filter.sh
# note that you need to change environment follow the instruction in run_filter.sh
```
## Evaluation
For human alignment test (tab1)
```
# clip_image clip_text clip_image_text lpips dino mse gpt my
python eval.py --metrics clip_image
```
For ground truth test (tab2)
```
python eval_gt_test.py
```
Evaluate position and size (tab4)
```
python eval_size_pos.py
```

# path for images and metadata
metadata for human alignment test
```
/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/metadata.json
/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/metadata_supp.json
```
metadata for ground truth test
```
/network_space/server128/shared/zhuoying/data/MyData/gt_test/gt_test_metadata.json
```
image path
```
/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test
```