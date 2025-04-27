from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn
import json
import torch
import argparse
import os
from tqdm import tqdm

def calculate_dino_similarity(image_path1, image_path2, model, processor, device):
    # 读取图像
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")
    
    # 处理图像并计算特征
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)

        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        outputs2 = model(**inputs2)
        image_features2 = outputs2.last_hidden_state
        image_features2 = image_features2.mean(dim=1)

    # 计算余弦相似度
    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0], image_features2[0]).item()
    sim = (sim + 1) / 2  # 将相似度归一化到 [0, 1] 范围内

    return sim

# model
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

# 
parser = argparse.ArgumentParser()
parser.add_argument("--method", default="ip2p", type=str)

args = parser.parse_args()

method = args.method
input_dir = '/network_space/server128/shared/zhuoying/data/MyData/user_study'

# 读取metadata.json文件
metadata_path = '/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/metadata.json'
with open(metadata_path, 'r') as f:
    data = json.load(f)
print('metadata loaded')

for item in tqdm(data):
    src_path = os.path.join(input_dir, item['source_image_path'])
    edited_path = os.path.join(input_dir, item['image_path'])
    score = calculate_dino_similarity(src_path, edited_path, model, processor, device)
    item['dino'] = score

# with open(metadata_path, 'w') as f:
#     json.dump(data, f, indent=4)