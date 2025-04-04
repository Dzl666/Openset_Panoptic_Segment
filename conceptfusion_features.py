import os
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
# segmentation
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# image embedding
import clip
from transformers import AutoProcessor, AutoModel, AutoTokenizer

from utils import *

torch.cuda.set_device(7)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'






def create_segmentor(
        model_name='sam2.1_hiera_large.pt', 
        cfg_name='sam2.1_hiera_l.yaml', 
        points_per_side=32,
        min_area=100):

    # The path is relative to the sam2 package
    model_cfg = f"configs/sam2.1/{cfg_name}"
    # This path is related to this code's location
    sam2_checkpoint = f"/home/zdeng/sam2/checkpoints/{model_name}"
    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, 
        device=DEVICE, apply_postprocessing=False
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=256,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.9,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_area,
        use_m2m=True,
    )
    return mask_generator


def main():

    torch.autograd.set_grad_enabled(False)

    seq_name = 'scene0001_00'
    data_dir = '/scratch/zdeng/datasets/scannet'
    # data_dir = '/cluster/scratch/dengzi/scannet'
    # image is 1296 * 968
    rgb_path = pjoin(data_dir, seq_name, 'color')
    file_names = os.listdir(rgb_path)
    file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

    f_n = file_names[5]
    rgb = cv2.imread(pjoin(rgb_path, f_n))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape

    # clip_model, prep_clip = create_feature_extractor('ViT-L/14', device=DEVICE)
    # clip_model.eval()

    # siglip_model = "google/siglip2-base-patch16-224"
    siglip_model = "google/siglip2-large-patch16-256"
    model = AutoModel.from_pretrained(siglip_model)
    tokenizer = AutoTokenizer.from_pretrained(siglip_model)
    processor = AutoProcessor.from_pretrained(siglip_model)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    mask_generator = create_segmentor(
        points_per_side=32, min_area=(H*W) * 0.0005,
    )
    
    masks = mask_generator.generate(rgb)
    show_anns(masks, rgb, f"sam_vis_{f_n}")
    
    global_feat = None
    # with torch.amp.autocast():
    print("Extracting global CLIP features...")
    _img = prep_clip(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    global_feat = clip_model.encode_image(_img)
    global_feat /= global_feat.norm(dim=-1, keepdim=True)

    global_feat = global_feat.half().to(DEVICE)
    global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # (1, 1024)
    feat_dim = global_feat.shape[-1]
    cos_sim_func = torch.nn.CosineSimilarity(dim=-1)

    feat_per_roi = []
    roi_inds = []
    sim_scores = []
    print("Extracting local CLIP features...")
    for m_id in range(len(masks)):
        mask_i = masks[m_id]
        _x, _y, _w, _h = tuple(mask_i["bbox"])  # xywh bounding box
        x1, x2 = int(_x), int(_x+_w)
        y1, y2 = int(_y), int(_y+_h)
        seg = mask_i["segmentation"]
        nonzero_inds = torch.argwhere(torch.from_numpy(seg))

        img_roi = rgb[y1:y2, x1:x2]
        img_roi = prep_clip(Image.fromarray(img_roi)).unsqueeze(0).to(DEVICE)
        roifeat = clip_model.encode_image(img_roi)
        roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
        feat_per_roi.append(roifeat)
        roi_inds.append(nonzero_inds)

    feat_per_roi = torch.cat(feat_per_roi)  # (N, 1024)
    # calculate all pairs cos_sim
    feat_per_roi_temp1 = feat_per_roi.unsqueeze(0)  # (1, N, 1024)
    feat_per_roi_temp2 = feat_per_roi_temp1.permute(1, 0, 2) # (N, 1, 1024)

    cross_roi_sim = cos_sim_func(feat_per_roi_temp1, feat_per_roi_temp2)  # (N, N)
    # sum up every row except the value in the diagonal
    self_roi_sim = torch.diagonal(cross_roi_sim, 0)  # (N, )
    cross_roi_sim -= torch.diag_embed(self_roi_sim)  # (N, N)
    cross_roi_sim = cross_roi_sim.mean(dim=1)  # (N, )
    glo2local_sim = cos_sim_func(global_feat, feat_per_roi) # (N,)

    sim_scores = cross_roi_sim + glo2local_sim
    softmax_scores = torch.nn.functional.softmax(sim_scores, dim=0)

    print("Fusing the CLIP features...")
    outfeat = torch.zeros(H, W, feat_dim, dtype=torch.half)
    for m_id in range(len(masks)):
        weight_fuse = softmax_scores[m_id]
        _weighted_feat = weight_fuse * global_feat + (1 - weight_fuse) * feat_per_roi[m_id]
        _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)

        outfeat[roi_inds[m_id][:, 0], roi_inds[m_id][:, 1]] += _weighted_feat[0].detach().cpu().half()
        outfeat[roi_inds[m_id][:, 0], roi_inds[m_id][:, 1]] = torch.nn.functional.normalize(
            outfeat[roi_inds[m_id][:, 0], roi_inds[m_id][:, 1]].float(), dim=-1
        ).half()

    outfeat = outfeat.unsqueeze(0).float()
    # outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
    # outfeat = torch.nn.functional.interpolate(outfeat, [args.desired_height, args.desired_width], mode="nearest")
    # outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
    outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
    outfeat = outfeat[0].half() # --> (H, W, feat_dim)




    text_cands = [
        'cabinet', 'bed', 'chair', 'truck', 'sofa', 'table', 'door', 
        'window', 'bookshelf', 'picture', 'desk', 'curtain', 'pillow',  
        'nightstand', 'toilet', 'sink', 'lamp', 
        'wall', 'floor', 'blinds', 'shelves', 'dresser', 'mirror', 
        'floor mat', 'clothes', 'ceiling', 'books', 'paper', 'towel', 
        'box', 'whiteboard', 
        'chair_leg', 'table_leg', 'chair_arm', 'sofa_arm', 'sofa_back', 
    ]
    text_tokens = clip.tokenize(text_cands).to(DEVICE)

    disp_img = torch.zeros(H, W, 3).float()
    per_pixel_sim = torch.zeros(H, W, len(text_cands)).float()
    outfeat = outfeat.float().cuda()

    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1) # (N, 1024)
        # text_feats = text_feats.unsqueeze(0).unsqueeze(0)  # (1, 1, 77, 1024)

        for i in range(len(text_cands)):
            text_feat = text_feats[i]
            _sim = cos_sim_func(outfeat, text_feat.unsqueeze(0).unsqueeze(0))
            per_pixel_sim[:, :, i] = _sim.squeeze(-1)

        # get the best match id for each pixel
        best_match = torch.argmax(per_pixel_sim, dim=-1)  # (H, W)


    pallete = get_new_pallete(len(text_cands))
    for _i in range(len(text_cands)):
        disp_img[best_match == _i] = pallete[_i]

    disp_img = 0.5 * disp_img.detach().cpu().numpy() + 0.5 * rgb.astype('float32') / 255.0

    fig = plt.figure(figsize=(15, 15))
    plt.imshow(disp_img)
    

    # torch.save(outfeat.detach().cpu(), savefile)

    # put the label on the sam result based on the best match
    for m_id in range(len(masks)):
        mask_i = masks[m_id]
        seg = mask_i["segmentation"]
        _x, _y, _w, _h = tuple(mask_i["bbox"])
        x1, x2 = int(_x), int(_x+_w)
        y1, y2 = int(_y), int(_y+_h)
        # vote for the best match
        best_match_i = best_match[seg]
        best_match_i = torch.bincount(best_match_i.flatten().long())
        best_match_i = torch.argmax(best_match_i).item()

        plt.text(_x+_w/2, _y+_h/2, f'{text_cands[best_match_i]}', fontsize=14)

    plt.legend(handles=[
        mpatches.Patch(
            color=(pallete[i][0].item(), 
                pallete[i][1].item(), pallete[i][2].item(),),
            label=text_cands[i],
        )
        for i in range(len(text_cands))]
    )
    plt.savefig(f"seg_label_{f_n}")
        


if __name__ == "__main__":
    main()
