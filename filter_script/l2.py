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
    img1_array = np.array(img1).astype(np.float32)
    img2_array = np.array(img2).astype(np.float32)

    # Calculate the squared difference between the two images
    diff = img1_array - img2_array
    squared_diff = np.sqrt(np.mean(diff ** 2, axis=2))

    # Calculate the mean squared difference
    mse = np.mean(squared_diff)

    return mse

# Load the metadata
# parser = argparse.ArgumentParser()
# parser.add_argument("--method", default="ip2p", type=str)

# args = parser.parse_args()

# method = args.method
image_dir = '/network_space/server128/shared/zhuoying/data/MyData/user_study/'

# 读取metadata.json文件
metadata_path = '/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/metadata.json'
with open(metadata_path, 'r') as f:
    data = json.load(f)
print('metadata loaded')

for entry in tqdm(data):
    for i in range(1,5):
        img1_path = os.path.join(image_dir, entry[f'edited_image{i}']['image_path'])
        img2_path = os.path.join(image_dir, entry['source_image_path'])

        mse = l2_distance(img1_path, img2_path)
        entry[f'edited_image{i}']['mse'] = float(-mse) # 这里因为是越小越好所以改过来了

# Save the updated metadata
with open(metadata_path, 'w') as f:
    json.dump(data, f, indent=4)