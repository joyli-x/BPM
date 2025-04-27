export CUDA_VISIBLE_DEVICES=7
HF_ENDPOINT=https://hf-mirror.com python bpm.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image test_data/5.png \
  --output_dir "outputs/my_metrics_style" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "what if there was a glass of soda on the table." \
  --device "cuda"