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
```
# step 1: run LLM
./run_llm.sh
# step 2: run BPM
./run_metrics.sh
```

## Evaluation Results
```
python eval.py
```

## TODO
- [ ] Add demo code and visualization code
- [ ] Collect a more diverse global editing dataset
- [ ] Add supports for "directional object moving"

## Acknowledgements
This code is built upon [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [Gemma](https://huggingface.co/princeton-nlp/gemma-2-9b-it-SimPO). Thanks for their great work.