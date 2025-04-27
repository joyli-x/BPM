# import json

# json_file = '../metadata/mgie_metadata.json'
# with open(json_file) as f:
#     data = json.load(f)

# clip_score_list = []

# for entry in data:
#     clip_score_list.append(entry['dino'])

# print(f"max: {max(clip_score_list)/100}")
# print(f"min: {min(clip_score_list)/100}")

# calculate the mse for two image
import json
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm

def l2_distance(img1_path, img2_path):
    resize_const = 500

    # Load the images
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # resize the image
    img1 = img1.resize((resize_const, resize_const))
    img2 = img2.resize((resize_const, resize_const))

    # Convert the images to numpy arrays
    img1_array = np.array(img1)
    # array中数字的类型
    print(img1_array.dtype)
    assert 0
    img2_array = np.array(img2)

    # Calculate the squared difference between the two images
    diff = img1_array - img2_array
    squared_diff = diff ** 2

    # Calculate the mean squared difference
    mse = np.mean(squared_diff)

    return mse

img1 = '/network_space/server128/shared/zhuoying/data/MyData/images/fake/DALLE_2/1.png'
img2 = '/network_space/server128/shared/zhuoying/data/MyData/images/fake/DALLE_2/2.png'
print(l2_distance(img1, img2))