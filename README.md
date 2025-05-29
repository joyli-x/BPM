# Balancing Preservation and Modification: A Region and Semantic-Aware Metric for Instruction-Based Image Editing
This is the official implementation of the paper: "Balancing Preservation and Modification: A Region and Semantic-Aware Metric for Instruction-Based Image Editing" (Accepted to ICML 2025)

## Installation
```
conda create -n bpm python=3.8
pip install -r requirements.txt
```
Then please follow [Grounded_Segment_Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main?tab=readme-ov-file#running_man-grounded-sam-detect-and-segment-everything-with-text-prompt) to download pretrained weights

## Download metrics evaluation dataset with human annotations
Please download the dataset from [Google Drive](https://drive.google.com/drive/folders/12Z4vX8pAGMbq7TtSTuGRCnHmvfb359XL?usp=drive_link), the `data` should be as follows:
```
data
├── sample_test (human align test images)
├── gt_test_images (ground truth test images)
├── local_metadata.json
├── global_metadata.json
├── gt_test_metadata.json
```

## Run BPM
```shell
# step 1: run LLM
python gemma.py

# step 2: run BPM
export CUDA_VISIBLE_DEVICES=0
HF_ENDPOINT=https://hf-mirror.com python bpm.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --output_dir "outputs/" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --device "cuda"
```

## Evaluation
```shell
# human alignment test (tab.1 and tab.2 in the paper)
python eval_human_alignment.py

# gt test (tab.4 in the paper)
python eval_gt_test.py
```

## TODO
- [ ] Add demo
- [ ] Collect a more diverse global editing dataset
- [ ] Add supports for "directional object moving"

## Acknowledgements
This code is built upon [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [Gemma](https://huggingface.co/princeton-nlp/gemma-2-9b-it-SimPO). Thanks for their great work.

## Citation
If you find this work useful, please consider citing:
```bibtex
TODO (upload to arxiv)
```
