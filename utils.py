import os, time, argparse, logging, pickle
from tqdm import tqdm
import cv2 #, imageio
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

# segmentation
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# feature embedder
import clip

def create_clip_extractor(model_name='ViT-B/32', device='cuda'):
    """
    - model_name: choose from ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 
        'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    """
    logging.info("Loading the CLIP model...")
    if model_name == 'ViT-B/32':
        model_path = '/scratch/zdeng/checkpoints/clip/ViT-B-32.pt'
    elif model_name == 'ViT-L/14':
        model_path = '/scratch/zdeng/checkpoints/clip/ViT-L-14.pt'
    else:
        raise ValueError(f"Unsupported CLIP model name: {model_name}")
    model, preprocess = clip.load(model_path, device=device)

    return model, preprocess


def create_sam_segmentor(
        model_name='sam2.1_hiera_large.pt',
        cfg_name='sam2.1_hiera_l.yaml',
        points_per_side=32,
        min_area=100, 
        iou_thres=0.7, 
        post_process=False, 
        device='cuda'
    ):

    if device == 'cuda':
        # use bfloat16
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # The path is relative to the sam2 package
    model_cfg = f"configs/sam2.1/{cfg_name}"
    # This path is related to this code's location
    sam2_checkpoint = f"/scratch/zdeng/checkpoints/{model_name}"
    sam2 = build_sam2(
        model_cfg, sam2_checkpoint,
        device=device, apply_postprocessing=post_process
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=256,
        pred_iou_thresh=iou_thres,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_area,
        use_m2m=True,
    )
    return mask_generator


def cos_sim(a, b):
    """
    Compute cosine similarity between two tensors.
    a: tensor of shape (N, D)
    b: tensor of shape (M, D)
    """
    dot_prod = np.dot(a, b.T)
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    cos_sim = dot_prod / (norm_a * norm_b.T)
    return cos_sim

def L1_sim(a, b):
    """
    Compute L1 similarity between two tensors.
    a: tensor of shape (N, D)
    b: tensor of shape (M, D)
    """
    a = np.expand_dims(a, axis=1) # (N, 1, D)
    b = np.expand_dims(b, axis=0) # (1, M, D)
    l1_sim = np.abs(a - b).mean(axis=-1) # (N, M)
    return -l1_sim


def get_new_pallete(num_colors):
    """Generate a color pallete given the number of colors needed. First color is always black."""
    pallete = []
    for j in range(num_colors):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        pallete.append([r, g, b])
    return torch.tensor(pallete).float() / 255.0


def show_anns(anns, rgb, save_path='./mask_map.png', borders=True):
    
    color_mask = np.ones((rgb.shape[0], rgb.shape[1], 4))
    color_mask[:, :, 3] = 0 # set all color mask to 0 alpha
    
    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    ax = fig.add_subplot(121)
    ax.axis('off')
    ax.imshow(rgb)

    for ann in anns:
        mask_area = ann['segmentation']
        color_mask[mask_area] = np.concatenate([np.random.random(3), [0.7]])

        sample_pt = ann['point_coords']
        score = ann['predicted_iou']
        bbox = ann['bbox']
        # ax.text(bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, f'{score:.3f}')

        if borders:
            contours, _ = cv2.findContours(mask_area.astype(np.uint8), 
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) \
                for contour in contours]
            cv2.drawContours(color_mask, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(color_mask)

    ax = fig.add_subplot(122)
    ax.imshow(rgb)
    ax.axis('off')

    plt.savefig(save_path)
    plt.close()