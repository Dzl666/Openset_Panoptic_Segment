import os, argparse, logging, json
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import supervision as sv

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import Model

# segment anything
from segment_anything import build_sam, SamPredictor
# Recognize Anything Model
from ram.models import ram, ram_plus, tag2text
from ram import inference_tag2text, inference_ram, get_transform
from ram.utils import build_openset_llm_label_embedding, build_openset_label_embedding

# self packages

torch.cuda.set_device(7)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')




def load_grounding_dino(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
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
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


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
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=1)) 
    ax.text(x0+w/2, y0+h/2, label, color='white')


def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'tags_chinese': tags_chinese,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)


def main():
    
    tag2text_ckpt = '/scratch/zdeng/checkpoints/tag2text_swin_14m.pth'
    ram_ckpt = '/scratch/zdeng/checkpoints/ram_swin_large_14m.pth'
    ram_plus_ckpt = '/scratch/zdeng/checkpoints/ram_plus_swin_large_14m.pth'
    # ram_plus_tags = '/home/zdeng/Openset_Panoptic_Segment/openimages_rare_200_llm_tag_descriptions.json'

    ground_dino_cfgs = './GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    ground_dino_ckpt = './GroundingDINO/weights/groundingdino_swint_ogc.pth'
    sam_ckpt = '/scratch/zdeng/checkpoints/sam_vit_h_4b8939.pth'

    seq_name = 'scene0000_00'
    data_dir = '/scratch/zdeng/datasets/scannet'
    output_dir = pjoin('results','ram_grounded_sam', seq_name)
    os.makedirs(output_dir, exist_ok=True)


    # image is 1296 * 968
    rgb_path = pjoin(data_dir, seq_name, 'color')
    file_names = os.listdir(rgb_path)
    file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

    # get H and W
    rgb = cv2.imread(pjoin(rgb_path, file_names[0]))
    H, W, _ = rgb.shape

    # parameteres
    box_threshold = 0.15
    text_threshold = 0.15
    # ## For NMS (Non-Maximum Suppression)
    iou_threshold = 0.5

    device = DEVICE
    
    image_path = pjoin(rgb_path, file_names[100])

    crop_size = 384
    # delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]
    # ram_model = tag2text(
    #     pretrained=tag2text_ckpt, image_size=crop_size, 
    #     vit='swin_b', delete_tag_index=delete_tag_index
    # )
    # ram_model.threshold = 0.68 # thres for tagging
    # ram_model = ram(
    #     pretrained=ram_ckpt, image_size=crop_size, vit='swin_l'
    # )
    ram_model = ram_plus(
        pretrained=ram_plus_ckpt, image_size=crop_size, vit='swin_l'
    )

    ram_model.eval()
    ram_model = ram_model.to(device)

    # grounding_dino_model = Model(model_config_path=ground_dino_cfgs, model_checkpoint_path=ground_dino_ckpt)
    grounding_dino_model = load_grounding_dino(ground_dino_cfgs, ground_dino_ckpt, device=device)

    sam_predictor = SamPredictor(build_sam(checkpoint=sam_ckpt).to(device))

    # ========================== RAM ==========================
    logging.info("start")

    # print('Building tag embedding:')
    # with open(ram_plus_tags, 'rb') as fo:
    #     ram_plus_tags = json.load(fo)
    # openset_label_embedding, openset_categories = build_openset_llm_label_embedding(ram_plus_tags)

    # openset_label_embedding, openset_categories = build_openset_label_embedding()
    # print(len(openset_label_embedding), len(openset_categories))
    # ram_model.tag_list = np.array(openset_categories)
    # ram_model.label_embed = nn.Parameter(openset_label_embedding.float())
    # ram_model.num_class = len(openset_categories)
    # # the threshold for unseen categories is often lower
    # ram_model.class_threshold = torch.ones(ram_model.num_class) * 0.5
    #######

    ram_transform = get_transform(image_size=crop_size)
    i_tensor = ram_transform(Image.open(image_path)).to(device)

    # ram_res = inference_tag2text(i_tensor.unsqueeze(0), ram_model)[0]
    ram_res = inference_ram(i_tensor.unsqueeze(0), ram_model)[0] # for both ram and ram_plus

    tags = ram_res.replace(' | ', ', ') #.split(', ')
    print("Image Tags: ", tags)

    
    
    # draw output image
    fig = plt.figure(figsize=(16, 8))
    ax0 = fig.add_subplot(121)
    ax0.imshow(image)
    ax0.axis('off')
    # for mask in masks:
    #     show_mask(mask, ax0.gca(), random_color=True)
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box, ax0.gca(), label)
    

    ax1 = fig.add_subplot(122)
    ax1.imshow(image)
    ax1.axis('off')

    plt.savefig(
        os.path.join(output_dir, 'sam_'+os.path.basename(image_path))
    )
    plt.close()


    

if __name__ == "__main__":
    main()
