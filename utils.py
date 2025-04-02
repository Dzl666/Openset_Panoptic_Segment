import os, time, argparse, logging, pickle
from tqdm import tqdm
import cv2 #, imageio
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

import clip

def create_feature_extractor(model_name='ViT-B/32', device='cuda'):
    """
    - model_name: choose from ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 
        'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    """
    logging.info("Loading the CLIP model...")
    model, preprocess = clip.load(model_name, device=device)

    return model, preprocess


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