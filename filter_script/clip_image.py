import json
import os
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from tqdm import tqdm
from PIL import ImageDraw, ImageFont
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="ip2p", type=str)

args = parser.parse_args()

method = args.method
input_dir = '../images'

# 读取metadata.json文件
metadata_path = f'../metadata/{method}_wo_style.json'
with open(metadata_path, 'r') as f:
    data = json.load(f)
print('metadata loaded')

# 创建sample_test文件夹
output_dir = f'../sample_test_clip_image/{method}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = CLIPModel.from_pretrained("/network_space/server128/shared/zhuoying/models/clip-vit-large-patch14")
preprocess = CLIPImageProcessor.from_pretrained("/network_space/server128/shared/zhuoying/models/clip-vit-large-patch14")
print('CLIP model loaded')

# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)
    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")
    # Return the preprocessed image
    return image


def clip_img_score(img1_path, img2_path):
    # Load the two images and preprocess them for CLIP
    image_a = load_and_preprocess_image(img1_path)["pixel_values"]
    image_b = load_and_preprocess_image(img2_path)["pixel_values"]

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)

    # Delete the image data to free up memory
    del image_a
    del image_b
    return similarity_score.item()


def save_comparison_image(src_path, edited_path, editing_instruction, output_path):
    # 绘制对比的图片
    img1 = Image.open(src_path).convert("RGB")
    img2 = Image.open(edited_path).convert("RGB")

    text_size = 20

    # 创建一个新的图片，大小为两张图片的宽度之和和高度的最大值
    new_img = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height) + 2 * text_size), "white")

    # 将两张图片粘贴到新的图片上
    new_img.paste(img1, (0, 2 * text_size))
    new_img.paste(img2, (img1.width, 2 * text_size))

    # 创建一个Draw对象
    draw = ImageDraw.Draw(new_img)

    # 加载字体
    font = ImageFont.truetype('/usr/share/fonts/truetype/tlwg/TlwgMono.ttf', text_size)  # 使用特定字体及大小

    # 在图片的上方添加instruction text
    text_top = 0
    draw.text((0, text_top), editing_instruction, font=font, fill='black')

    # 保存合成的图片
    new_img.save(output_path)

    del img1
    del img2

my_try = 0

for item in tqdm(data):
    image_path1 = os.path.join(input_dir, item['source_image_path'])
    image_path2 = os.path.join(input_dir, item['image_path'])
    score = clip_img_score(image_path1, image_path2)
    item['clip_image'] = score
    # print(score)

# # save
# with open(metadata_path, 'w') as f:
#     json.dump(data, f, indent=4)


# 根据CLIP score进行排序，选择得分最高的前30个图像
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
