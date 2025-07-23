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


def main():
    torch.autograd.set_grad_enabled(False)

    seq_name = 'scene0000_00'
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

    load_from_pkl = False
    encoder = 'siglip' # 'clip' or 'siglip'
    fuse_method = 'patch-seg' # 'glo-seg' or 'patch-seg'


    exp_name = f'{fuse_method}_{encoder}'
    result_dir = pjoin('results', seq_name)
    os.makedirs(pjoin(result_dir, 'sam_temp_res'), exist_ok=True)
    os.makedirs(pjoin(result_dir, 'fusion_vis', exp_name), exist_ok=True)
    os.makedirs(pjoin(result_dir, 'openseg_fusion'), exist_ok=True)



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
    # text_dict = {}
    # for i, text in enumerate(text_cands):
    #     text_dict[text] = text_feats[i]
    # save_pkl_path = pjoin(result_dir, 'siglip2_text_embed.pkl')
    # with open(save_pkl_path, "wb") as f:
    #     pickle.dump(text_dict, f)


    if not load_from_pkl:
        segmentor = create_sam_segmentor(
            points_per_side=32, min_area=(H*W) * 0.0005, 
            iou_thres=0.75, post_process=True, device=DEVICE
        )

    cos_sim_func = torch.nn.CosineSimilarity(dim=-1)
    
    for f_idx in tqdm(range(0, num_frame, step)):
        rgb_file = pjoin(rgb_path, file_names[f_idx])
        rgb = cv2.imread(rgb_file)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # ========================= SAM =========================
        if load_from_pkl:
            with open(pjoin(result_dir, 'sam_temp_res', 
                f'{f_idx}_post.pkl'), "rb") as f:
                mask_dict = pickle.load(f)
        else:
            with torch.no_grad():
                mask_dict = segmentor.generate(rgb)
            if len(mask_dict) == 0:
                logging.warning("No mask found, skipping..")
                continue

            mask_dict = sorted(mask_dict, key=(lambda x: x['area']), reverse=True)

            # save the intermidiate results
            with open(pjoin(result_dir, 'sam_temp_res', 
                f'{f_idx}_post.pkl'), "wb") as f:
                pickle.dump(mask_dict, f)


        # ## Get global feature
        if fuse_method == 'glo-seg':
            if encoder == 'clip':
                _img = prep_clip(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    global_feat = clip_model.encode_image(_img)
            elif encoder == 'siglip':
                img_inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    global_feat = siglip_model.get_image_features(**img_inputs)
            global_feat = torch.nn.functional.normalize(global_feat, dim=-1).half()


        # #### Get patch features in 4 layers
        if fuse_method == 'patch-seg':
            patch_num = [(6, 8), (4, 6), (3, 4), (2, 3)] # (H,W)
            patch_maps = torch.zeros((4, H, W), dtype=torch.int16)
            patch_area = [] # should from small to large
            patch_feat_list = []

            for i, (p_num_h, p_num_w) in enumerate(patch_num):
                patch_h = H // p_num_h
                patch_w = W // p_num_w
                patch_area.append(patch_h * patch_w)

                patch_feats = []
                for j in range(p_num_h):
                    for k in range(p_num_w):
                        y1, y2 = j*patch_h, (j+1)*patch_h
                        x1, x2 = k*patch_w, (k+1)*patch_w
                        patch_maps[i, y1:y2, x1:x2] = j * p_num_w + k
                        patch = Image.fromarray(rgb[y1:y2, x1:x2])

                        if encoder == 'siglip':
                            img_inputs = processor(images=patch, return_tensors="pt").to(DEVICE)
                            with torch.no_grad():
                                patchfeat = siglip_model.get_image_features(**img_inputs)
                            patchfeat = torch.nn.functional.normalize(patchfeat, dim=-1)
                        patch_feats.append(patchfeat)
                patch_feats = torch.cat(patch_feats).half()  # (num_patches, 1024)
                patch_feat_list.append(patch_feats)
            # print(f"Patch area of all layers: {patch_area})


        # ## Get per-segment features
        feat_per_roi = []
        roi_inds = []
        roi_area = []
        num_segs = len(mask_dict)
        for m_id in range(num_segs):
            mask_i = mask_dict[m_id]
            _x, _y, _w, _h = tuple(mask_i["bbox"])  # xywh bounding box
            roi_area.append(_w * _h)
            x1, x2 = int(_x), int(_x+_w)
            y1, y2 = int(_y), int(_y+_h)
            seg = mask_i["segmentation"] # each of them is non-overlapping
            nonzero_inds = torch.argwhere(torch.from_numpy(seg))

            # ##### mask out not relevant area
            valid_mask = (seg > 0)
            img_roi = copy.deepcopy(rgb)
            img_roi[~valid_mask] = np.array([255, 255, 255]) # set to white
            img_roi = Image.fromarray(img_roi[y1:y2, x1:x2])

            if encoder == 'clip':
                img_roi = prep_clip(img_roi).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    roifeat = clip_model.encode_image(img_roi)
            elif encoder == 'siglip':
                img_inputs = processor(images=img_roi, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    roifeat = siglip_model.get_image_features(**img_inputs)
            
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_inds.append(nonzero_inds)

        feat_per_roi = torch.cat(feat_per_roi).half()  # (N, 1024)



        # ================== Fusing all features ==================
        # print("Fusing the CLIP features...")
        label_map = torch.zeros((H, W), dtype=torch.uint8)
        feat_map = torch.zeros((num_segs, feat_dim)).half()

        if fuse_method == 'glo-seg':
            glo2local_sim = cos_sim_func(global_feat, feat_per_roi) # (N,)
            # # calculate all pairs cos_sim
            # feat_per_roi_temp1 = feat_per_roi.unsqueeze(0)  # (1, N, 1024)
            # feat_per_roi_temp2 = feat_per_roi_temp1.permute(1, 0, 2) # (N, 1, 1024)

            # cross_roi_sim = cos_sim_func(feat_per_roi_temp1, feat_per_roi_temp2)  # (N, N)
            # # sum up every row except the value in the diagonal
            # self_roi_sim = torch.diagonal(cross_roi_sim, 0)  # (N, )
            # cross_roi_sim -= torch.diag_embed(self_roi_sim)  # (N, N)
            # cross_roi_sim = cross_roi_sim.mean(dim=1)  # (N, )
            # sim_scores = cross_roi_sim + glo2local_sim
            sim_scores = glo2local_sim
            softmax_scores = torch.nn.functional.softmax(sim_scores, dim=0)

            for m_id in range(num_segs):
                roi_idx = roi_inds[m_id]
                label_map[roi_idx[:, 0], roi_idx[:, 1]] = m_id+1

                weight_fuse = softmax_scores[m_id]
                _weighted_feat = weight_fuse * global_feat + (1 - weight_fuse) * feat_per_roi[m_id]
                _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
                feat_map[m_id] = _weighted_feat

        if fuse_method == 'patch-seg':
            import bisect
            for m_id in range(num_segs):

                roi_idx = roi_inds[m_id]
                bbox_area = roi_area[m_id]
                label_map[roi_idx[:, 0], roi_idx[:, 1]] = m_id+1

                # find the patch set of closer and larger area in patch_size
                layer_idx = bisect.bisect_right(patch_area, len(roi_idx))
                layer_idx = np.clip(layer_idx, 0, len(patch_num)-1)

                patch_mask = patch_maps[layer_idx, roi_idx[:, 0], roi_idx[:, 1]].detach().cpu().numpy()
                unique_values, counts = np.unique(patch_mask, return_counts=True)

                # print(f"Layer {layer_idx}, patch area: {patch_area[layer_idx]}, seg area: {len(roi_idx)}, bbox: {bbox_area}")
                
                # only get the feat of the path with > 30% overlapping
                patch_area_thres = patch_area[layer_idx] * 0.3
                patch_idx = unique_values[counts > patch_area_thres]
                
                feats_patch = patch_feat_list[layer_idx][patch_idx] # (M, 1024)
                seg2patch_sim = cos_sim_func(feats_patch, feat_per_roi[m_id]) # (M, ) in range (-1, 1)
                # drop those with negative similarity
                valid_feat = (seg2patch_sim > 0.0)
                if seg2patch_sim.shape[0] > 0:
                    max_sim = torch.max(seg2patch_sim, dim=0).values.item()
                else:
                    max_sim = 0.0
                
                # TODO what should we do to the cross attention to relevant patches
                weight_fuse = torch.nn.functional.softmax(seg2patch_sim[valid_feat], dim=0) # (M, ) in range (0, 1)

                _weighted_feat = max_sim * torch.sum(weight_fuse.unsqueeze(1) * feats_patch[valid_feat], dim=0) + feat_per_roi[m_id]
                _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
                feat_map[m_id] = _weighted_feat


        feat_map = feat_map.float().cuda() # (num_segs, feat_dim)

        # find the best match to the text candidates
        per_seg_sim = cos_sim_func(feat_map.unsqueeze(1), text_feats.unsqueeze(0))  # (num_segs, num_text_cands)
        best_matches = torch.argmax(per_seg_sim, dim=-1)  # (num_segs, )



        # ================== Visualization ==================
        seg_dict = {}
        seg_meta_info = {}


        fig = plt.figure(figsize=(30, 15))

        ax0 = fig.add_subplot(121)
        ax0.axis('off')
        ax0.imshow(rgb)

        feat_img = torch.zeros(H, W, 4).float()
        seg_img = np.zeros((H, W, 4), dtype=np.float32)
        for m_id in range(num_segs):
            best_match_seg = best_matches[m_id]
            color = torch.cat([pallete[best_match_seg], torch.tensor([0.5])])
            feat_img[label_map == (m_id+1)] = color

            inst_info = {}
            
            mask_i = mask_dict[m_id]
            seg = mask_i["segmentation"]
            seg_img[seg] = np.concatenate([np.random.random(3), [0.2]])

            _x, _y, _w, _h = tuple(mask_i["bbox"])  # xywh bounding box
            x1, x2 = int(_x), int(_x+_w)
            y1, y2 = int(_y), int(_y+_h)
            inst_info['bbox'] = [x1, y1, x2, y2]
            inst_info['score_iou'] = mask_i['predicted_iou']
            inst_info['area'] = len(roi_inds[m_id])

            inst_info['sem_feat'] = feat_map[m_id].detach().cpu().numpy() 
            match_text = text_cands[best_match_seg]
            inst_info['text_idx'] = match_text
            seg_meta_info[m_id+1] = inst_info

            plt.text(_x+_w/2, _y+_h/2, match_text, fontsize=14, color='white')
        
        seg_dict['seg_map'] = label_map.detach().cpu().numpy()
        seg_dict['meta_info'] = seg_meta_info


        feat_img = feat_img.numpy()
        feat_img[:, :, 3] = 0.3
        ax0.imshow(feat_img)
        # let the legend out side the image to the right
        ax0.legend(handles=[
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

        ax1 = fig.add_subplot(122)
        ax1.axis('off')
        ax1.imshow(rgb)
        ax1.imshow(seg_img)
        plt.savefig(pjoin(result_dir, 'fusion_vis', exp_name, f'{f_idx}.png'),
                    bbox_inches='tight')
        plt.close(fig)

        # ### save the segmentation results
        save_pkl_path = pjoin(result_dir, 'openseg_fusion', str(f_idx).zfill(5)+'.pkl')
        with open(save_pkl_path, "wb") as f:
            pickle.dump(seg_dict, f)



if __name__ == "__main__":
    main()
