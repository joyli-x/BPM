import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def clip_text_score(image_path, text):
    # 预处理图像
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
        
    # 处理输入
    inputs = clip_processor(
        images=image,
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # 获取图像和文本特征
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
    # 归一化特征向量
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # 计算余弦相似度
    similarity = torch.cosine_similarity(image_features, text_features)
    
    return similarity.item()

# clip_score 函数
def clip_score(src_image, tar_image, src_txt, trg_txt):
    
    # 处理图像和文本
    inputs_src = clip_processor(text=[src_txt], images=src_image, return_tensors="pt", padding=True).to(device)
    # print(inputs_src.keys())
    inputs_tar = clip_processor(text=[trg_txt], images=tar_image, return_tensors="pt", padding=True).to(device)
    
    # 编码图像和文本特征
    with torch.no_grad():
        src_img_features = clip_model.get_image_features(inputs_src['pixel_values'])
        tar_img_features = clip_model.get_image_features(inputs_tar['pixel_values'])
        src_txt_features = clip_model.get_text_features(inputs_src['input_ids'], inputs_src['attention_mask'])
        trg_txt_features = clip_model.get_text_features(inputs_tar['input_ids'], inputs_tar['attention_mask'])
    
    # 计算图像和文本特征的差值
    if "None" in src_txt:
        delta_text = trg_txt_features
    elif "None" in trg_txt:
        delta_text = -src_txt_features
    else:
        delta_text = trg_txt_features - src_txt_features
    delta_img = tar_img_features - src_img_features
    
    # 计算余弦相似度 (即CLIPScore)
    score = torch.cosine_similarity(delta_img, delta_text)
    
    return score.item()

def expand_box(boxes, scale=1.2, image_shape=None):
    # 如果只有一个box，直接使用它
    if len(boxes) == 1:
        box = boxes[0]
    else:
        # 计算能装得下所有box的最小包围框
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        box = [x0, y0, x1, y1]

    # 计算原框的中心
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    
    # 计算新框的尺寸
    width = box[2] - box[0]
    height = box[3] - box[1]
    
    new_width = width * scale
    new_height = height * scale
    
    # 计算新的框的左上角和右下角
    new_x0 = center_x - new_width / 2
    new_y0 = center_y - new_height / 2
    new_x1 = center_x + new_width / 2
    new_y1 = center_y + new_height / 2
    
    # 确保新框不会超出图片边界
    if image_shape is not None:
        height, width, _ = image_shape
        new_x0 = max(0, new_x0)
        new_y0 = max(0, new_y0)
        new_x1 = min(width, new_x1)
        new_y1 = min(height, new_y1)
        
    return np.array([new_x0, new_y0, new_x1, new_y1])

def crop_box(image, box):
    # 将box用作裁剪框，确保box的值是整数
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])
    
    # 裁剪并返回新图像
    return image[y0:y1, x0:x1]

def calculate_l2_distance(image_path1, image_path2, mask, shape_const=500):
    # 读取图像
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # 确保两张图像的大小相同
    img1 = cv2.resize(img1, (shape_const, shape_const))
    img2 = cv2.resize(img2, (shape_const, shape_const))

    # 计算l2距离
    # 转换为float32
    # img1 = img1.astype(np.float32)
    # img2 = img2.astype(np.float32)
    diff = img1 - img2
    l2_distance = np.sqrt(np.mean(diff ** 2, axis=2))

    # 应用掩码
    masked_distances = l2_distance[mask == 0]

    # 计算平均值
    if len(masked_distances) > 0:
        average_l2_distance = np.mean(masked_distances)
    else:
        average_l2_distance = 0.0  # 如果没有掩码区域，返回0

    return float(average_l2_distance / 255) # 返回归一化的值



from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load sam model
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    # load clip model
    clip_model = CLIPModel.from_pretrained("/network_space/server128/shared/zhuoying/models/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("/network_space/server128/shared/zhuoying/models/clip-vit-large-patch14")
    print('CLIP model loaded')

    # load json
    # json_file = '/network_space/server128/shared/zhuoying/data/MyData/metadata/dalle2_metadata.json'
    # with open(json_file, 'r') as f:
    #     metadata = json.load(f)
    
    # load idx
    idx_file = '/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/metadata.json'
    # idx_list = []
    with open(idx_file, 'r') as f:
        user_data = json.load(f)
    # for entry in user_data:
    #     idx_list.append(entry['idx'])
    
    image_dir = '/network_space/server128/shared/zhuoying/data/MyData/user_study/'
    
    image_resize_const = 500
    process_cnt = 0
    cnt = 0

    # idx2model = {'1': 'mgie', '2': 'ft_ip2p', '3': 'ip2p', '4': 'dalle2', '5': 'genartist', '6': 'ace'}
    idx2model = {'1': 'pie', '2': 'hq_edit', '3': 'ft_ip2p', '4': 'ip2p'}
    LLM_model = 'gemma'

    for item in tqdm(user_data):
        for image_idx in idx2model:
            entry = item[f'edited_image{image_idx}']
            # entry['source_image_path'] = os.path.join('sample_test/real/ms_coco/', os.path.basename(entry['image_path']))
            entry['source_image_path'] = os.path.join(image_dir, item['source_image_path'])
            entry['original_image_edited_area'] = item['LLM'][f'{LLM_model}_origin']
            entry['edited_image_edited_area'] = item['LLM'][f'{LLM_model}_edited']
            if "added_object" in entry:
                entry['edited_image_edited_area'] = item[f'added_object']
            
            if "all" in entry['original_image_edited_area']:
                item[f'edited_image{image_idx}'][f'my_edit_quality_{LLM_model}'] = clip_text_score(os.path.join(image_dir, entry['image_path']), item['instruction'])
                item[f'edited_image{image_idx}'][f'my_preservation_{LLM_model}'] = 0
                continue

            modification_score = 0
            consistency_score = 0
            masks_list = []

            # try:
            # get bbox
            if "None" in entry['original_image_edited_area']:
                image_path = os.path.join(image_dir, entry['image_path'])
                text_prompt = entry['edited_image_edited_area'] # added_object??

                # load image
                image_pil, image = load_image(image_path)

                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold, text_threshold, device=device
                )

                # 没有检测到bbox
                if len(pred_phrases) == 0:
                    modification_score = 0

            else:
                image_path = os.path.join(image_dir, entry['source_image_path'])
                text_prompt = entry['original_image_edited_area']
            
                # load image
                image_pil, image = load_image(image_path)

                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold, text_threshold, device=device
                )

                # 没有检测到bbox
                if len(pred_phrases) == 0:
                    item[f'edited_image{image_idx}'][f'my_edit_quality_{LLM_model}'] = clip_text_score(os.path.join(image_dir, entry['image_path']), entry['edited_image_edited_area'])
                    item[f'edited_image{image_idx}'][f'my_preservation_{LLM_model}'] = 0
                    # print("error: no bbox in original image")
                    # print(f"{entry['image_path']}")
                    cnt += 1
                    continue

            if len(pred_phrases) != 0:
                # 加载修改后的图
                image_path = os.path.join(image_dir, entry['image_path'])
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (image_resize_const, image_resize_const))
                # predictor.set_image(image)

                # 加载原图
                source_image_path = os.path.join(image_dir, entry['source_image_path'])
                source_image = cv2.imread(source_image_path)
                source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
                source_image = cv2.resize(source_image, (image_resize_const, image_resize_const))

                size = (image_resize_const, image_resize_const)
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                boxes_filt = boxes_filt.cpu()

                # 扩大box并裁剪图像
                expanded_box = expand_box(boxes_filt.numpy(), scale=1.5, image_shape=image.shape)
                cropped_image = crop_box(image, expanded_box)
                cropped_source_image = crop_box(source_image, expanded_box)

                # # 保存裁剪后的图像
                # cropped_image_pil = Image.fromarray(cropped_image)
                # save_path = os.path.join(output_dir, 'cropped_image', entry['image_path'])
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # cropped_image_pil.save(save_path)

                # modification score: clip text
                modification_score = clip_score(cropped_source_image, cropped_image, entry['original_image_edited_area'], entry['edited_image_edited_area']) # added_object??
                # modification_score = clip_text_score(cropped_image,entry['edited_image_edited_area'])

                # consistency score: l2
                for j in range(2):
                    if j % 2 == 0:
                        # original image
                        if "None" in entry['original_image_edited_area']:
                            continue
                        image_path = os.path.join(image_dir, entry['source_image_path'])
                    else:
                        # edited image
                        if "None" in entry['edited_image_edited_area']:
                            continue
                        image_path = os.path.join(image_dir, entry['image_path'])
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (image_resize_const, image_resize_const))

                    predictor.set_image(image)
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

                    masks, _, _ = predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes.to(device),
                        multimask_output = False,
                    )
                    masks_list.extend(masks)

                # merge masks
                if len(masks_list) == 0:
                    mask_img = torch.zeros((image_resize_const, image_resize_const))
                else:
                    mask_img = torch.zeros(masks_list[0].shape[-2:])
                    for mask in masks_list:
                        mask_img[mask.cpu().numpy()[0] == True] = 1

                # save mask
                save_path = os.path.join(output_dir, 'mask', entry['image_path'])
                # print(save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, mask_img.numpy()*255)

                consistency_score = calculate_l2_distance(os.path.join(image_dir, entry['source_image_path']), os.path.join(image_dir, entry['image_path']), mask_img)
            
            else:
                consistency_score = calculate_l2_distance(os.path.join(image_dir, entry['source_image_path']), os.path.join(image_dir, entry['image_path']), torch.zeros((image_resize_const, image_resize_const)))
            
            # save data
            item[f'edited_image{image_idx}'][f'my_edit_quality_{LLM_model}'] = modification_score
            item[f'edited_image{image_idx}'][f'my_preservation_{LLM_model}'] = -consistency_score
            # item[f'edited_image{image_idx}']['my_dalle2_consistency_score'] = consistency_score
            # user_data[i]['my_mgie_score'] = modification_score - consistency_score


            # except Exception as e:
            #     print(f"error: {entry['image_path']}")
            #     print(f"error: {e}")
            #     continue

            # if process_cnt <= 30:
            # draw output image
            # plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.imread(os.path.join(image_dir, entry['source_image_path'])))
            # # for mask in masks:
            # #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            # for box, label in zip(boxes_filt, pred_phrases):
            #     show_box(box.numpy(), plt.gca(), label)

            # plt.axis('off')
            # save_path = os.path.join(output_dir.replace('outputs', 'vis_result'), entry['image_path'])
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # plt.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
            # plt.close()
            
            process_cnt += 1

        # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
    # print(f"zero bbox: {cnt}")
    # print(f"process_cnt: {process_cnt}")
    with open(idx_file, 'w') as f:
        json.dump(user_data, f, indent=4)
