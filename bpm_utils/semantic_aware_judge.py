# Semantic-Aware Judge

import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F

# clip text
def clip_text_score(image_path, text, clip_processor, clip_model, device):
    # read image
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
        
    # preprocess input
    inputs = clip_processor(
        images=image,
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # encode image and text
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
    # normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # compute cosine similarity
    similarity = torch.cosine_similarity(image_features, text_features)
    
    return similarity.item()

# directional clip similarity
def clip_score(src_image, tar_image, src_txt, trg_txt, clip_processor, clip_model, device):
    # preprocess image and text
    inputs_src = clip_processor(text=[src_txt], images=src_image, return_tensors="pt", padding=True).to(device)
    # print(inputs_src.keys())
    inputs_tar = clip_processor(text=[trg_txt], images=tar_image, return_tensors="pt", padding=True).to(device)
    
    # encode image and tect
    with torch.no_grad():
        src_img_features = clip_model.get_image_features(inputs_src['pixel_values'])
        tar_img_features = clip_model.get_image_features(inputs_tar['pixel_values'])
        src_txt_features = clip_model.get_text_features(inputs_src['input_ids'], inputs_src['attention_mask'])
        trg_txt_features = clip_model.get_text_features(inputs_tar['input_ids'], inputs_tar['attention_mask'])
    
    # calculate delta_text and delta_img
    if "none" in src_txt.lower():
        delta_text = trg_txt_features
    elif "none" in trg_txt.lower():
        delta_text = -src_txt_features
    else:
        delta_text = trg_txt_features - src_txt_features
    delta_img = tar_img_features - src_img_features
    
    score = torch.cosine_similarity(delta_img, delta_text)
    
    return score.item()

def expand_box(boxes, scale=1.2, image_shape=None):
    # if there is only one box, use it directly
    if len(boxes) == 1:
        box = boxes[0]
    else:
        # compute the minimum enclosing box that can contain all boxes
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        box = [x0, y0, x1, y1]

    # calculate the center of the original box
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    
    # calculate the width and height of the original box
    width = box[2] - box[0]
    height = box[3] - box[1]
    
    new_width = width * scale
    new_height = height * scale
    
    # calculate the new box coordinates
    new_x0 = center_x - new_width / 2
    new_y0 = center_y - new_height / 2
    new_x1 = center_x + new_width / 2
    new_y1 = center_y + new_height / 2
    
    # make sure the new box does not exceed the image boundaries
    if image_shape is not None:
        height, width, _ = image_shape
        new_x0 = max(0, new_x0)
        new_y0 = max(0, new_y0)
        new_x1 = min(width, new_x1)
        new_y1 = min(height, new_y1)
        
    return np.array([new_x0, new_y0, new_x1, new_y1])

def crop_box(image, box):
    # use box as the cropping box, ensure the values of box are integers
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])
    
    # crop the image using the box
    return image[y0:y1, x0:x1]

def calculate_l2_distance(image_path1, image_path2, mask, shape_const=500):
    # read image
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # make sure the images are in the same shape
    img1 = cv2.resize(img1, (shape_const, shape_const))
    img2 = cv2.resize(img2, (shape_const, shape_const))

    # l2 distance
    diff = img1 - img2
    l2_distance = np.sqrt(np.mean(diff ** 2, axis=2))

    # apply mask
    masked_distances = l2_distance[mask == 0]

    # calculate average l2 distance
    if len(masked_distances) > 0:
        average_l2_distance = np.mean(masked_distances)
    else:
        average_l2_distance = 0.0  # if no pixels are masked, set to 0

    return float(average_l2_distance / 255) # return normalized distance