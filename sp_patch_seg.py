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
    num_frame = 2 * step

    encoder = 'clip' # 'clip' or 'siglip'

    exp_name = f'sp_{encoder}'
    result_dir = pjoin('results', seq_name)
    os.makedirs(pjoin(result_dir, 'sp_vis', exp_name), exist_ok=True)
    os.makedirs(pjoin(result_dir, 'openseg_sp'), exist_ok=True)



    if encoder == 'clip':
        # feat_dim = 512 # for ViT-B/32
        feat_dim = 768 # for ViT-L/14
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
        
    text_feats = text_feats.detach().cpu() #.to(torch.float32).numpy()

    cos_sim_func = torch.nn.CosineSimilarity(dim=-1)

    # crop the whole image into multiple scale of patches
    crop_size = [(H, W), (H//2, W//2), (H//4, W//4)]
    clip_p_size = 14
    clip_ptk_num = 16
    # each patch is 14 x 14 cropped from the 224 x 224 image after downsampling
    patch_map = np.zeros((clip_p_size*clip_ptk_num, clip_p_size*clip_ptk_num), dtype=np.uint8)
    for patch_idx in range(clip_ptk_num * clip_ptk_num):
        _row = (patch_idx // clip_ptk_num)
        _col = patch_idx - _row * clip_ptk_num
        _x = _col * clip_p_size
        _y = _row * clip_p_size
        # set the patch map to 1
        patch_map[_y:_y+clip_p_size, _x:_x+clip_p_size] = patch_idx
    

    for f_idx in tqdm(range(0, num_frame, step)):
        rgb_file = pjoin(rgb_path, file_names[f_idx])
        rgb = cv2.imread(rgb_file)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        feat_map = np.zeros((H, W, feat_dim), dtype=np.float32)

        for scale_layer in range(1):
            crop_h, crop_w = crop_size[scale_layer]

            layer_feat_map = np.zeros_like(feat_map, dtype=np.float32)
            map_cnt = np.zeros((H, W), dtype=np.int16)

            # use sliding window to crop the image
            stride_h = int(crop_h // 2)
            stride_w = int(crop_w // 2)
            for i in range(0, H - crop_h + 1, stride_h):
                for j in range(0, W - crop_w + 1, stride_w):
                    # crop the image
                    rgb_patch = rgb[i:i + crop_h, j:j + crop_w, :]
                    cls_tk, ptks = clip_model.encode_image(prep_clip(Image.fromarray(rgb_patch)).unsqueeze(0).to(DEVICE))
                    ptks = ptks.squeeze().detach().cpu().numpy()
                    

                    lab_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2LAB)
                    slic_segs = segmentation.slic(lab_patch, n_segments=50, compactness=10, sigma=2)
                    seg_idx, seg_area = np.unique(slic_segs, return_counts=True)

                    resize_patch_map = cv2.resize(patch_map, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                    patch_size = crop_w* crop_h / clip_ptk_num / clip_ptk_num

                    feat_crop = np.zeros((crop_h, crop_w, feat_dim), dtype=np.float32)
                    # find the idx of patches covered by each slis seg
                    for slic_id in seg_idx:
                        patch_cover = resize_patch_map[slic_segs == slic_id]
                        patch_idx, patch_area = np.unique(patch_cover, return_counts=True)
                        # remove the id with area < 10% of each patch
                        patch_idx = patch_idx[patch_area > 0.1 * patch_size]
                        patch_feat = np.mean(ptks[patch_idx], axis=0)
                        feat_crop[slic_segs == slic_id] = patch_feat

                    layer_feat_map[i:i + crop_h, j:j + crop_w, :] += feat_crop
                    map_cnt[i:i + crop_h, j:j + crop_w] += 1

            map_cnt[map_cnt == 0] = 1
            layer_feat_map /= map_cnt[:, :, np.newaxis].astype(np.float32)
            feat_map += layer_feat_map
        
        # feat_map /= len(crop_size)
        feat_map = torch.from_numpy(feat_map)

        print("Visualizing the segmentation results")
        per_pixel_sim = torch.zeros((H, W, len(text_cands)), dtype=torch.float32)
        for i in range(len(text_cands)):
            text_feat = text_feats[i]
            _sim = cos_sim_func(feat_map, text_feat.unsqueeze(0).unsqueeze(0))
            per_pixel_sim[:, :, i] = _sim.squeeze(-1)

        # get the best match id for each pixel
        best_match = torch.argmax(per_pixel_sim, dim=-1)  # (H, W)

        disp_img = torch.zeros((H, W, 3)).float()
        pallete = get_new_pallete(len(text_cands))
        for _i in range(len(text_cands)):
            disp_img[best_match == _i] = pallete[_i]
    
        disp_img = 0.5 * disp_img.detach().cpu().numpy() + 0.5 * rgb.astype('float32') / 255.0
    
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(disp_img)
        plt.axis('off')
        plt.legend(handles=[
            mpatches.Patch(
                color=(pallete[i][0].item(),
                    pallete[i][1].item(), pallete[i][2].item(),),
                label=text_cands[i],
            )
            for i in range(len(text_cands))], 
            loc='upper left', 
            bbox_to_anchor=(1.05, 1), 
            borderaxespad=0.0,
        )
        plt.savefig(f"results/sp_seg_{f_idx}.png")

        # ### save the segmentation results
        # save_pkl_path = pjoin(result_dir, 'openseg_sp', str(f_idx).zfill(5)+'.pkl')
        # with open(save_pkl_path, "wb") as f:
        #     pickle.dump(seg_dict, f)



if __name__ == "__main__":
    main()
