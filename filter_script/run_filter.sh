source /network_space/server128/shared/zhuoying/anaconda3/bin/activate
conda activate ip2p
python clip_text.py
echo "clip-text done"
python l2.py
echo "l2 done"
python run_lpips.py
echo "lpips done"

source /network_space/server128/shared/zhuoying/anaconda3/bin/activate
conda activate mgie
HF_ENDPOINT=https://hf-mirror.com python run_dino.py
echo "dino done"
python clip_image_new.py
echo "clip-image done"
python clip_image_text.py
echo "clip-image-text done"