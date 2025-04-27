import json
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from PIL import ImageDraw, ImageFont
import argparse
import torch
import torch.nn.functional as F

# def get_clip_score(image_path, text):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits_per_image = outputs.logits_per_image
#     # Delete the image data to free up memory
#     del image
#     return logits_per_image.item()

def calculate_clip_similarity(image_path, text):
    """
    计算图像和文本之间的CLIP相似度分数
    
    参数:
        image_path: 图像文件的路径或PIL Image对象
        text: 要比较的文本字符串
    
    返回:
        float: 余弦相似度分数 (-1 到 1 之间)
    """
    
    # 预处理图像
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
        
    # 处理输入
    inputs = processor(
        images=image,
        text=[text],
        return_tensors="pt",
        padding=True
    )
    
    # 获取图像和文本特征
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
    # 归一化特征向量
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # 计算余弦相似度
    similarity = torch.cosine_similarity(image_features, text_features)
    
    return similarity.item()


input_dir = '/network_space/server128/shared/zhuoying/data/MyData/user_study'

# 读取metadata.json文件
metadata_path = f'/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/metadata.json'
with open(metadata_path, 'r') as f:
    data = json.load(f)
print('metadata loaded')

# model
model = CLIPModel.from_pretrained("/network_space/server128/shared/zhuoying/models/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("/network_space/server128/shared/zhuoying/models/clip-vit-large-patch14")
print('CLIP model loaded')
my_try = 0

# 计算每张图像和指令的CLIP score
scores = []
for item in tqdm(data):
    for i in range(1,5):
        image_path = os.path.join(input_dir, item[f"edited_image{i}"]['image_path'])
        image = Image.open(image_path).convert("RGB")
        instruction = item['instruction']
        score = calculate_clip_similarity(image_path, instruction)
        # instruction = item[f"edited_image{i}"]['edited_image_edited_area']
        # if "None" in instruction:
        #     instruction = item[f"edited_image{i}"]['original_image_edited_area']
        #     score = 1-calculate_clip_similarity(image_path, instruction)
        # else:
        #     score = calculate_clip_similarity(image_path, instruction)
        item[f"edited_image{i}"]['clip_text'] = score
    # print(score)

# save
with open(metadata_path, 'w') as f:
    json.dump(data, f, indent=4)

# # 根据CLIP score进行排序，选择得分最高的前30个图像
# scores.sort(key=lambda x: x[1], reverse=True)
# top_30 = scores[:30]


# cnt = 0

# # 将得分最高的30张图像复制到sample_test文件夹中
# for item, score in top_30:
#     cnt += 1
#     src_path = os.path.join(input_dir, item['source_image_path'])
#     edited_path = os.path.join(input_dir, item['image_path'])
#     output_path = os.path.join(output_dir, f'{cnt}.png')
#     editing_instruction = f"{item['instruction']}_score:{score}"
#     save_comparison_image(src_path, edited_path, editing_instruction, output_path)
