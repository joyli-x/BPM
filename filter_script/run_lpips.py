import json
import os
from PIL import Image
from tqdm import tqdm
from PIL import ImageDraw, ImageFont
import argparse
import lpips
import torchvision.transforms as transforms


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

# # 创建sample_test文件夹
# output_dir = f'../sample_test_lpips/{method}'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

loss_fn = lpips.LPIPS(net='alex')

# 加载图片
def load_image(image_path, size=(512, 512)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)  # Resize the image
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# 计算 LPIPS 距离
def calculate_lpips(img1_path, img2_path, loss_fn):
    # 加载图片
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # 计算 LPIPS 距离
    distance = loss_fn(img1, img2)

    # clean up memory
    del img1
    del img2
    
    return distance.item()


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

# 计算每张图像的score
scores = []
for item in tqdm(data):
    src_path = os.path.join(input_dir, item['source_image_path'])
    edited_path = os.path.join(input_dir, item['image_path'])
    score = calculate_lpips(src_path, edited_path, loss_fn)
    item['lpips'] = score

# with open(metadata_path, 'w') as f:
#     json.dump(data, f, indent=4)
# # 根据Lpips进行排序，选择得分最低的前30个图像
# scores.sort(key=lambda x: x[1], reverse=False)
# top_30 = scores[:30]

# cnt = 0

# # 将得分最低的30张图像复制到sample_test文件夹中
# for item, score in top_30:
#     cnt += 1
#     src_path = os.path.join(input_dir, item['source_image_path'])
#     edited_path = os.path.join(input_dir, item['image_path'])
#     output_path = os.path.join(output_dir, f'{cnt}.png')
#     editing_instruction = f"{item['instruction']}_score:{score}"
#     save_comparison_image(src_path, edited_path, editing_instruction, output_path)
