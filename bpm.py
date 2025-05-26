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

# BPM
from bpm_utils.region_aware_judge import determine_size, determine_position_multi_bbox
from bpm_utils.semantic_aware_judge import clip_text_score, clip_score, expand_box, crop_box, calculate_l2_distance


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
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print('CLIP model loaded')
    
    # load idx
    idx_file = './data/local_metadata.json'
    with open(idx_file, 'r') as f:
        user_data = json.load(f)
    
    image_dir = './data/'
    
    image_resize_const = 500
    process_cnt = 0
    cnt = 0

    idx2model = {'1': 'pie', '2': 'hq_edit', '3': 'ft_ip2p', '4': 'ip2p'}
    LLM_model = 'gemma'

    for item in tqdm(user_data):
        for image_idx in idx2model:
            entry = item[f'edited_image{image_idx}']
            entry['source_image_path'] = os.path.join(image_dir, item['source_image_path'])

            entry['original_image_edited_area'] = item['original_part']
            entry['edited_image_edited_area'] = item['edited_part']
            if 'none' in entry['original_image_edited_area'] and item['added_object']:
                entry['edited_image_edited_area'] = item['added_object']
            
            # global edit
            if "all" in entry['original_image_edited_area']:
                item[f'edited_image{image_idx}'][f'edit_quality'] = clip_text_score(
                                                                    os.path.join(image_dir, entry['image_path']), 
                                                                    item['instruction'], 
                                                                    clip_processor=clip_processor, 
                                                                    clip_model=clip_model, 
                                                                    device=device
                                                                    )
                item[f'edited_image{image_idx}'][f'preservation'] = 0
                item[f'edited_image{image_idx}'][f'size'] = 1
                item[f'edited_image{image_idx}'][f'position'] = 1
                continue

            # local edit
            modification_score = 0
            consistency_score = 0
            masks_list = []

            # detect bboxes in original image and edited image
            # source image
            source_boxes_filt = []
            if "none" not in entry['original_image_edited_area'] or item['being_added']:
                # load source image
                source_image_path = os.path.join(image_dir, entry['source_image_path'])
                if "none" in entry['original_image_edited_area']:
                    source_text_prompt = item['being_added']
                else:
                    source_text_prompt = entry['original_image_edited_area']
                # load image
                source_image_pil, source_image = load_image(source_image_path)

                # run grounding dino model
                source_boxes_filt, source_pred_phrases = get_grounding_output(
                    model, source_image, source_text_prompt, box_threshold, text_threshold, device=device
                )

                # resize bbox
                size = (image_resize_const, image_resize_const)
                H, W = size[1], size[0]
                for i in range(source_boxes_filt.size(0)):
                    source_boxes_filt[i] = source_boxes_filt[i] * torch.Tensor([W, H, W, H])
                    source_boxes_filt[i][:2] -= source_boxes_filt[i][2:] / 2
                    source_boxes_filt[i][2:] += source_boxes_filt[i][:2]
                source_boxes_filt = source_boxes_filt.cpu()

            # edited image
            edited_boxes_filt = []
            if "none" not in entry['edited_image_edited_area']:
                # load edited image
                edited_image_path = os.path.join(image_dir, entry['image_path'])
                edited_text_prompt = entry['edited_image_edited_area']
                # load image
                edited_image_pil, edited_image = load_image(edited_image_path)

                # run grounding dino model
                edited_boxes_filt, edited_pred_phrases = get_grounding_output(
                    model, edited_image, edited_text_prompt, box_threshold, text_threshold, device=device
                )

                # resize bbox
                size = (image_resize_const, image_resize_const)
                H, W = size[1], size[0]
                for i in range(edited_boxes_filt.size(0)):
                    edited_boxes_filt[i] = edited_boxes_filt[i] * torch.Tensor([W, H, W, H])
                    edited_boxes_filt[i][:2] -= edited_boxes_filt[i][2:] / 2
                    edited_boxes_filt[i][2:] += edited_boxes_filt[i][:2]
                edited_boxes_filt = edited_boxes_filt.cpu()
            
            # Region-Aware Judge
            item[f'edited_image{image_idx}'][f'size'] = float(determine_size(edited_boxes_filt, 
                                                                      source_boxes_filt,
                                                                      item))
            item[f'edited_image{image_idx}'][f'position'] = float(determine_position_multi_bbox(edited_boxes_filt, 
                                                                                         source_boxes_filt, 
                                                                                         item))

            # Semantic-Aware Judge
            # special case: no bbox detected in original image and it's not 'adding' (detection failed)
            if "none" not in entry['original_image_edited_area']:
                if len(source_pred_phrases) == 0:
                    modification_score = clip_text_score(
                        os.path.join(image_dir, entry['image_path']), 
                        entry['edited_image_edited_area'], 
                        clip_processor=clip_processor, 
                        clip_model=clip_model, 
                        device=device)
                    item[f'edited_image{image_idx}'][f'edit_quality'] = modification_score

                    consistency_score = calculate_l2_distance(os.path.join(image_dir, entry['source_image_path']), 
                                                          os.path.join(image_dir, entry['image_path']), 
                                                          torch.zeros((image_resize_const, image_resize_const)))
                    item[f'edited_image{image_idx}'][f'preservation'] = -consistency_score

                    print("error: no bbox detected in original image")
                    print(f"{entry['image_path']}")
                    cnt += 1
                    continue

            # special case: no bbox detected in edited image and it's not 'removing' (edit failed)
            if "none" not in entry['edited_image_edited_area']:
                if len(edited_pred_phrases) == 0:
                    modification_score = 0
                    item[f'edited_image{image_idx}'][f'edit_quality'] = modification_score

                    consistency_score = calculate_l2_distance(os.path.join(image_dir, entry['source_image_path']), 
                                                          os.path.join(image_dir, entry['image_path']), 
                                                          torch.zeros((image_resize_const, image_resize_const)))
                    item[f'edited_image{image_idx}'][f'preservation'] = -consistency_score
                    continue

            # normal case
            # load edited image
            image_path = os.path.join(image_dir, entry['image_path'])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_resize_const, image_resize_const))

            # load original image
            source_image_path = os.path.join(image_dir, entry['source_image_path'])
            source_image = cv2.imread(source_image_path)
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            source_image = cv2.resize(source_image, (image_resize_const, image_resize_const))

            # expand bbox and crop image
            if 'none' in entry['original_image_edited_area']: # add
                expanded_box = expand_box(edited_boxes_filt.numpy(), scale=1.5, image_shape=image.shape)
            else:
                expanded_box = expand_box(source_boxes_filt.numpy(), scale=1.5, image_shape=image.shape)
            cropped_image = crop_box(image, expanded_box)
            cropped_source_image = crop_box(source_image, expanded_box)

            # # save cropped image
            # cropped_image_pil = Image.fromarray(cropped_image)
            # save_path = os.path.join(output_dir, 'cropped_image', entry['image_path'])
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # cropped_image_pil.save(save_path)

            # modification score: directional clip similarity
            modification_score = clip_score(cropped_source_image, 
                                            cropped_image, 
                                            entry['original_image_edited_area'], 
                                            entry['edited_image_edited_area'],
                                            clip_processor=clip_processor,
                                            clip_model=clip_model,
                                            device=device
                                            )
            item[f'edited_image{image_idx}'][f'edit_quality'] = modification_score
                
            # consistency score: l2
            # detect masks
            for j in range(2):
                if j % 2 == 0:
                    # original image
                    if "none" in entry['original_image_edited_area']:
                        continue
                    predictor.set_image(source_image)
                else:
                    # edited image
                    if "none" in entry['edited_image_edited_area']:
                        continue
                    predictor.set_image(image)

                # detect masks
                if "none" in entry['original_image_edited_area']:
                    transformed_boxes = predictor.transform.apply_boxes_torch(edited_boxes_filt, image.shape[:2]).to(device)
                else:
                    transformed_boxes = predictor.transform.apply_boxes_torch(source_boxes_filt, image.shape[:2]).to(device)

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
            # save_path = os.path.join(output_dir, 'mask', entry['image_path'])
            # # print(save_path)
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # cv2.imwrite(save_path, mask_img.numpy()*255)

            consistency_score = calculate_l2_distance(os.path.join(image_dir, entry['source_image_path']), 
                                                        os.path.join(image_dir, entry['image_path']), 
                                                        mask_img)
            item[f'edited_image{image_idx}'][f'preservation'] = -consistency_score
            
            process_cnt += 1

        # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
    print(f"zero bbox: {cnt}")
    print(f"process_cnt: {process_cnt}")
    with open(idx_file, 'w') as f:
        json.dump(user_data, f, indent=4)
