import os, logging, copy
import pickle as pkl
from os.path import join as pjoin
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# EXTERNAL
# image embedding
import clip
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from skimage import segmentation, data, color

from utils import *

torch.cuda.set_device(7)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

text_cands = [
    'cabinet', 'bed', 'chair', 'truck', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'desk', 'curtain', 'pillow',  
    'nightstand', 'toilet', 'sink', 'lamp',
    'wall', 'floor', 'blinds', 'shelves', 'mirror',
    'floor mat', 'clothes', 'ceiling', 'books', 'paper', 'towel',
    'box', 'whiteboard',
    'chair_leg', 'sofa_arm', 'sofa_back', 'table_leg', 'chair_arm',
]
pallete = get_new_pallete(len(text_cands))


def test_sp():
    seq_name = 'scene0001_00'
    data_dir = '/scratch/zdeng/datasets/scannet'
    rgb_path = pjoin(data_dir, seq_name, 'color')
    file_names = os.listdir(rgb_path)

    rgb0 = cv2.imread(pjoin(rgb_path, file_names[0]))
    rgb0 = rgb0.astype('float32') / 255.0

    # take 1/4 of the image
    height, width, _ = rgb0.shape
    rgb0 = rgb0[int(height / 4):int(height * 3 / 4), int(width / 4):int(width * 3 / 4), :]

    rgb0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2RGB)
    # convert to LAB
    lab = cv2.cvtColor(rgb0, cv2.COLOR_RGB2LAB)
    segments = segmentation.slic(lab, n_segments=150, compactness=10, sigma=1)
    boundary_mask = segmentation.mark_boundaries(rgb0, segments, color=(1, 0, 0))

    alpha = 0.6  # Transparency level
    overlay = rgb0 * (1 - alpha) + boundary_mask * alpha

    fig, ax = plt.subplots()
    ax.imshow(overlay)
    plt.savefig('slic.png')


def main():
    torch.autograd.set_grad_enabled(False)

    seq_name = 'scene0001_00'
    data_dir = '/scratch/zdeng/datasets/scannet'
    # data_dir = '/cluster/scratch/dengzi/scannet'
    # image is 1296 * 968
    rgb_path = pjoin(data_dir, seq_name, 'color')
    file_names = os.listdir(rgb_path)
    file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

    # get H and W
    rgb = cv2.imread(pjoin(rgb_path, file_names[0]))
    H, W, _ = rgb.shape

    num_frame = len(file_names)
    step = 5
    num_frame = 40 * step

    encoder = 'siglip' # 'clip' or 'siglip'

    exp_name = f'sp_{encoder}'
    result_dir = pjoin('results', seq_name)
    os.makedirs(pjoin(result_dir, 'sp_vis', exp_name), exist_ok=True)
    os.makedirs(pjoin(result_dir, 'openseg_sp'), exist_ok=True)



    if encoder == 'clip':
        # feat_dim = 512 # for ViT-B/32
        # feat_dim = 768 # for ViT-L/14
        clip_model, prep_clip = create_clip_extractor('ViT-L/14', device=DEVICE)
        text_inputs = clip.tokenize(text_cands, context_length=77).to(DEVICE)
        with torch.no_grad():
            text_feats = clip_model.encode_text(text_inputs)
    elif encoder == 'siglip':
        feat_dim = 1152
        # siglip_model_name = "google/siglip2-base-patch16-224"
        siglip_model_path = '/scratch/zdeng/checkpoints/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f/'
        siglip_model = AutoModel.from_pretrained(siglip_model_path).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(siglip_model_path)
        processor = AutoProcessor.from_pretrained(siglip_model_path, use_fast=True)

        text_inputs = tokenizer(text_cands, 
            padding="max_length", max_length=64, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            text_feats = siglip_model.get_text_features(**text_inputs)
        
    # text_feats = text_feats.detach().cpu().to(torch.float32).numpy()


    cos_sim_func = torch.nn.CosineSimilarity(dim=-1)
    
    for f_idx in tqdm(range(0, num_frame, step)):
        rgb_file = pjoin(rgb_path, file_names[f_idx])
        rgb = cv2.imread(rgb_file)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

        slic_segs = segmentation.slic(lab, n_segments=150, compactness=10, sigma=1)

        # ### save the segmentation results
        # save_pkl_path = pjoin(result_dir, 'openseg_sp', str(f_idx).zfill(5)+'.pkl')
        # with open(save_pkl_path, "wb") as f:
        #     pickle.dump(seg_dict, f)



if __name__ == "__main__":
    main()
