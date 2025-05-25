# Balancing Preservation and Modification: A Region and Semantic-Aware Metric for Instruction-Based Image Editing
This is the official implementation of the paper: "Balancing Preservation and Modification: A Region and Semantic-Aware Metric for Instruction-Based Image Editing" (Accepted to ICML 2025)

## Installation
```
conda create -n bpm python=3.8
pip install -r requirements.txt
```
Then please follow [Grounded_Segment_Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main?tab=readme-ov-file#running_man-grounded-sam-detect-and-segment-everything-with-text-prompt) to download pretrained weights

## Download metrics evaluation dataset
Please download the dataset from [Google Drive](https://drive.google.com/file/d/1q0g2r7vX3x4a5b6c8j9kz1f8e5g3h4d/view?usp=sharing) and unzip it to the path './data/', the data should be as follows:
```
data
├── sample_test (images)
├── local_metadata.json
├── global_metadata.json
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

## Evaluation Results
```shell
python eval.py
```

## TODO
- [ ] Add demo code and visualization code
- [ ] Collect a more diverse global editing dataset
- [ ] Add supports for "directional object moving"

## Acknowledgements
This code is built upon [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [Gemma](https://huggingface.co/princeton-nlp/gemma-2-9b-it-SimPO). Thanks for their great work.