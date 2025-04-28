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

def determine_size(bbox1, bbox2, relation, threshold1=0.8, threshold2=1.2):
    '''- 变大：after_bbox_size / before > thresh1
    # - 变小：after_bbox_size / before < thresh2
    # - unchanged 在thresh2和thresh1之间'''

    before_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    after_size = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    ratio = after_size / before_size

    if relation == 'bigger':
        return ratio > threshold2
    elif relation == 'smaller':
        return ratio < threshold1
    elif relation == 'unchanged':
        return threshold1 <= ratio <= threshold2
    else:
        raise ValueError('relation should be larger, smaller or unchanged')



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

    idx2model = {'1': 'mgie', '2': 'ft_ip2p', '3': 'ip2p', '4': 'dalle2'}
    LLM_model = 'gemma'

    for item in tqdm(user_data):
        for image_idx in idx2model:
            entry = item[f'edited_image{image_idx}']
            entry['source_image_path'] = os.path.join('sample_test/real/ms_coco/', os.path.basename(entry['image_path']))
            entry['original_image_edited_area'] = item['LLM'][f'{LLM_model}_origin']
            entry['edited_image_edited_area'] = item['LLM'][f'{LLM_model}_edited']

            size_score = 0

            # add or remove: score=1
            if "None" in entry['original_image_edited_area'] or "None" in entry['edited_image_edited_area']:
                size_score = 1
            
            else:
                # 分别detect原图和edited图的bbox
                # source image
                source_image_path = os.path.join(image_dir, entry['source_image_path'])
                source_text_prompt = entry['original_image_edited_area']
                # load image
                source_image_pil, source_image = load_image(source_image_path)

                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    model, source_image, source_text_prompt, box_threshold, text_threshold, device=device
                )

                # resize bbox
                size = (image_resize_const, image_resize_const)
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                source_boxes_filt = boxes_filt.cpu()

                # draw output image
                plt.figure(figsize=(10, 10))
                source_image = cv2.imread(os.path.join(image_dir, entry['source_image_path']))
                source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
                source_image = cv2.resize(source_image, (image_resize_const, image_resize_const))
                plt.imshow(source_image)
                # for mask in masks:
                #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                for box, label in zip(source_boxes_filt, pred_phrases):
                    show_box(box.numpy(), plt.gca(), label)
                plt.axis('off')
                save_path = os.path.join(output_dir.replace('outputs', 'vis_result'), entry['source_image_path'])
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
                plt.close()


                # edited image
                edited_image_path = os.path.join(image_dir, entry['image_path'])
                edited_text_prompt = entry['edited_image_edited_area']
                # load image
                edited_image_pil, edited_image = load_image(edited_image_path)

                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    model, edited_image, edited_text_prompt, box_threshold, text_threshold, device=device
                )

                # resize bbox
                size = (image_resize_const, image_resize_const)
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]
                
                edited_boxes_filt = boxes_filt.cpu()

                # draw output image
                plt.figure(figsize=(10, 10))
                edited_image = cv2.imread(edited_image_path)
                edited_image = cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)
                edited_image = cv2.resize(edited_image, (image_resize_const, image_resize_const))
                plt.imshow(edited_image)
                # for mask in masks:
                #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                for box, label in zip(edited_boxes_filt, pred_phrases):
                    show_box(box.numpy(), plt.gca(), label)
                plt.axis('off')
                save_path = os.path.join(output_dir.replace('outputs', 'vis_result'), entry['image_path'])
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
                plt.close()

                # print(f"edited bbox: {edited_boxes_filt}")
                # print(f"source bbox: {source_boxes_filt}")
                # print(f"pred phrase: {pred_phrases}")

                if len(source_boxes_filt) == 0:
                    size_score = 1
                else:
                    if len(edited_boxes_filt) == 0:
                        size_score = 0
                    else:
                        size_score = determine_size(source_boxes_filt[0], edited_boxes_filt[0], 'unchanged', 0.7, 1.5).item()
            
            # save data
            item[f'edited_image{image_idx}'][f'my_size'] = int(size_score)
            
            
            process_cnt += 1

        # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
    print(f"zero bbox: {cnt}")
    print(f"process_cnt: {process_cnt}")
    with open(idx_file, 'w') as f:
        json.dump(user_data, f, indent=4)
