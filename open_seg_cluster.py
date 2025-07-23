import os, time, argparse, logging, pickle, copy
from os.path import join as pjoin
from tqdm import tqdm
import cv2 #, imageio
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# external libs
# segmentation
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# image embedding
import clip

from transformers import AutoProcessor, AutoModel, AutoTokenizer

# self packages
from utils import *

# 

# cabinet,bed,chair,truck,sofa,table,door,window,bookshelf,picture,desk,curtain,pillow,nightstand,toilet,sink,lamp,wall,floor,blinds,shelves,dresser,mirror,floor mat,clothes,ceiling,books,paper,towel,box,whiteboard,chair_leg,table_leg,chair_arm,sofa_arm,sofa_back



FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

torch.cuda.set_device(6)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    # use bfloat16
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# srun --time=0:5:00 -n 1 --mem-per-cpu=4g --gpus=1 --gres=gpumem:10g --pty bash
# module load stack/2024-06 && module load gcc/12.2.0 python/3.10.13 cuda/12.1.1 cudnn/8.9.7.29-12 cmake/3.27.7 eth_proxy
# source sam-env/bin/activate && cd Openset_Panoptic_Segment

# sbatch --time=0:5:00 -n 2 --mem-per-cpu=4g --gpus=1 --gres=gpumem:16g --output="logs/running.log" --wrap="python open_pano_seg.py"


def main():
    np.random.seed(3)
    print(f"======== using device: {DEVICE} ========")

    seq_name = 'scene0001_00'
    data_dir = '/scratch/zdeng/datasets/scannet'
    # data_dir = '/cluster/scratch/dengzi/scannet'
    # image is 1296 * 968
    rgb_path = pjoin(data_dir, seq_name, 'color')
    
    
    file_names = os.listdir(rgb_path)
    # strip the extensions and sort by the numeric part of the file names
    file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))
    num_frame = len(file_names)
    step = 5
    num_frame = 160 * step

    text_cands = [
        'cabinet', 'bed', 'chair', 'truck', 'sofa', 'table', 'door', 
        'window', 'bookshelf', 'picture', 'desk', 'curtain', 
        'pillow', 'nightstand', 'toilet', 'sink', 'lamp', 
        'wall', 'floor', 'blinds', 'shelves', 'dresser', 'mirror', 
        'floor mat', 'clothes', 'ceiling', 'books', 'paper', 'towel', 
        'box', 'whiteboard',
        'chair_leg', 'table_leg', 'chair_arm', 'sofa_arm', 'sofa_back', 
    ]
    pallete = get_new_pallete(len(text_cands))

    
    patch_num = (6, 4) # num_W, num_H | (8, 6) or (6, 4)
    division_method = 'seg' # 'patch' or 'seg'
    sim_metric = 'cos'
    encoder = 'clip' # 'clip' or 'siglip'

    exp_name = f'seg_{encoder}'
    result_dir = pjoin('results', seq_name)
    os.makedirs(pjoin(result_dir, 'sam_temp_res'), exist_ok=True)
    os.makedirs(pjoin(result_dir, 'clip_vis', exp_name), exist_ok=True)
    os.makedirs(pjoin(result_dir, f'openseg_{exp_name}'), exist_ok=True)

    
    if encoder == 'clip':
        feat_dim = 512 # for ViT-B/32
        # feat_dim = 768 # for ViT-L/14
        clip_model, prep_clip = create_clip_extractor('ViT-B/32', device=DEVICE)

        text_inputs = clip.tokenize(text_cands, context_length=77).to(DEVICE)
        with torch.no_grad():
            text_feats = clip_model.encode_text(text_inputs)

    elif encoder == 'siglip':
        feat_dim = 1152 # for siglip2-so400m-patch14-384
        # siglip_model_name = "google/siglip2-base-patch16-224"
        # siglip_model_name = "google/siglip2-so400m-patch14-384"
        siglip_model_path = '/scratch/zdeng/checkpoints/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f/'
        siglip_model = AutoModel.from_pretrained(siglip_model_path).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(siglip_model_path)
        processor = AutoProcessor.from_pretrained(siglip_model_path, use_fast=True)

        text_inputs = tokenizer(text_cands, 
            padding="max_length", max_length=64, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            text_feats = siglip_model.get_text_features(**text_inputs)
        
    text_feats = text_feats.detach().cpu().to(torch.float32).numpy()


    rgb_file = pjoin(rgb_path, file_names[0])
    rgb = cv2.imread(rgb_file)
    H, W, _ = rgb.shape

    # NOTE parameters
    load_from_pkl = True
    # create segmentor
    if not load_from_pkl:
        segmentor:SAM2AutomaticMaskGenerator = create_sam_segmentor(
            points_per_side=32, min_area=(H*W) * 0.0005, 
            iou_thres=0.75, post_process=True, device=DEVICE
        )

    # collect all features for later clustering
    all_feats = []
    patch_dict_list = []
    
    
    for f_idx in tqdm(range(0, num_frame, step)):
        # logging.info(f"Processing frame: {file_names[idx]}")
        
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
                logging.warning("No mask is segmented by sam2, skipping..")
                continue

            mask_dict = sorted(mask_dict, key=(lambda x: x['area']), reverse=True)

            # store it into files
            with open(pjoin(result_dir, 'sam_temp_res', 
                f'{f_idx}_post.pkl'), "wb") as f:
                pickle.dump(mask_dict, f)

            # show_anns(mask_dict, rgb, f"{result_dir}/sam_vis/{idx}_{num_sam_samples}samples_post.png")

        # logging.info(f"Total candidates: {len(mask_dict)}")



        # ============================ Feature Embedding ============================
        patch_dict = {}
        img_feats = []
        # ##### method 2 patch context
        if division_method == 'patch':
            # divide the rgb into N*M equal parts
            patch_size = (rgb.shape[1] // patch_num[0], rgb.shape[0] // patch_num[1])
            for i in range(patch_num[0]):
                for j in range(patch_num[1]):
                    x1, x2 = i * patch_size[0], (i+1) * patch_size[0]
                    y1, y2 = j * patch_size[1], (j+1) * patch_size[1]
                    cropped_rgb = rgb[y1:y2, x1:x2]

                    with torch.no_grad():
                        if encoder == 'clip':
                            image_feat_patch = clip_model.encode_image(
                                prep_clip(Image.fromarray(cropped_rgb)).unsqueeze(0).to(DEVICE))
                        elif encoder == 'siglip':
                            img_inputs = processor(images=Image.fromarray(cropped_rgb), 
                                return_tensors="pt").to(DEVICE)
                            image_feat_patch = siglip_model.get_image_features(**img_inputs)

                    image_feat_patch = image_feat_patch.detach().cpu().to(torch.float32).numpy()

                    img_feats.append(image_feat_patch)
                    patch_dict[(i, j)] = {
                        'range': [x1, y1, x2, y2],
                        'feat': image_feat_patch
                    }

        # ##### per segment context
        elif division_method == 'seg':
            for inst_id, ann in enumerate(mask_dict):
                bbox_seg = ann['bbox'] # XYWH format
                x1, x2 = int(bbox_seg[0]), int(bbox_seg[0]+bbox_seg[2])
                y1, y2 = int(bbox_seg[1]), int(bbox_seg[1]+bbox_seg[3])
                
                # mask out not relevant area
                mask_area = ann['segmentation']
                valid_mask = (mask_area > 0)
                img_roi = copy.deepcopy(rgb)
                img_roi[~valid_mask] = np.array([255, 255, 255]) # set to white
                img_roi = Image.fromarray(img_roi[y1:y2, x1:x2])

                with torch.no_grad():
                    if encoder == 'clip':
                        img_roi = prep_clip(img_roi).unsqueeze(0).to(DEVICE)
                        image_feat = clip_model.encode_image(img_roi)
                    elif encoder == 'siglip':
                        img_inputs = processor(images=img_roi, 
                            return_tensors="pt").to(DEVICE)
                        image_feat = siglip_model.get_image_features(**img_inputs)
                
                image_feat = image_feat.detach().cpu().to(torch.float32).numpy()
                img_feats.append(image_feat)
                patch_dict[inst_id] = {
                    'range': [x1, y1, x2, y2],
                    'feat': image_feat
                }   

        patch_dict_list.append(patch_dict)
        all_feats.append(np.array(img_feats))

        # =======================================================



    # find the closest text word for each patch 
    # based on the cosine similarity
    all_feats = np.concatenate(all_feats, axis=0)
    feat_dim = all_feats.shape[-1]
    all_feats = all_feats.reshape(-1, feat_dim)
    
    batch_size = 128
    iters = all_feats.shape[0] // batch_size + 1
    all_matches = []
    all_scores = []
    for i in range(iters):
        start = i * batch_size
        end = start+batch_size
        
        sim_score = cos_sim(all_feats[start:end], text_feats)

        # get top-3 idx and the scores
        # sort in descending order
        top3_idx = np.argsort(sim_score, axis=1)[:, -3:] # (B, 3)
        top3_score = np.take_along_axis(sim_score, top3_idx, axis=1)
        # text_idx = np.argmax(cos_sim, axis=1) # (B, )

        all_matches.append(top3_idx)
        all_scores.append(top3_score)

    all_matches = np.concatenate(all_matches, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    seg_iter = 0

    # collect the results and pack
    for iter, f_idx in tqdm(enumerate(range(0, num_frame, step))):

        with open(f"{result_dir}/sam_temp_res/{f_idx}_post.pkl", "rb") as f:
            mask_dict = pickle.load(f)

        rgb_file = pjoin(rgb_path, file_names[f_idx])
        rgb = cv2.imread(rgb_file)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(20, 10))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        ax = fig.add_subplot(121)
        ax.imshow(rgb)
        color_mask = np.ones((rgb.shape[0], rgb.shape[1], 4))
        color_mask[:, :, 3] = 0.0

        patch_dict = patch_dict_list[iter]

        # ###### method 2 patch context
        if division_method == 'patch':
            for i in range(patch_num[0]):
                for j in range(patch_num[1]):
                    x1, y1, x2, y2 = patch_dict[(i, j)]['range']
                    # color this patch
                    color_mask[y1:y2, x1:x2] = np.concatenate([np.random.random(3), [0.5]])
                    # 
                    top3_idx = all_matches[seg_iter]
                    top3_score = all_scores[seg_iter]
                    seg_iter += 1
                    patch_dict[(i, j)]['text_idx'] = top3_idx[-1]

                    ax.text(x1, y1*2/3 + y2/3, 
                        f'{text_cands[top3_idx[-1]]}:{top3_score[-1]:.2f}', 
                        fontsize=9, color='white')
                    ax.text(x1, y1/3 + y2*2/3, 
                        f'{text_cands[top3_idx[-2]]}:{top3_score[-2]:.2f}', 
                        fontsize=9, color='white')
                    ax.text(x1, y2, 
                        f'{text_cands[top3_idx[-3]]}:{top3_score[-3]:.2f}', 
                        fontsize=9, color='white')


        seg_dict = {}
        seg_meta_info = {}
        seg_map = np.zeros(rgb.shape[:2], dtype=np.uint8)

        # ##### Assign per segment info
        for inst_id, ann in enumerate(mask_dict):
            inst_info = {}
            bbox_seg = ann['bbox'] # XYWH format
            x1, x2 = int(bbox_seg[0]), int(bbox_seg[0]+bbox_seg[2])
            y1, y2 = int(bbox_seg[1]), int(bbox_seg[1]+bbox_seg[3])
            inst_info['bbox'] = [x1, y1, x2, y2]
            
            mask_area = ann['segmentation']
            seg_map[mask_area] = inst_id + 1

            # inst_info['area'] = np.count_nonzero(mask_area)
            # inst_info['score_iou'] = ann['predicted_iou']

            # ##### patch context, find the patch with largest overlap
            if division_method == 'patch':
                max_overlap = 0
                clip_feat = None
                text_idx = 0
                for key, patch_info in patch_dict.items():
                    x1, y1, x2, y2 = patch_info['range']
                    overlap = np.count_nonzero(mask_area[y1:y2, x1:x2])
                    if overlap > max_overlap:
                        max_overlap = overlap
                        clip_feat = patch_info['feat']
                        text_idx = patch_info['text_idx']

            # ##### per segment context
            elif division_method == 'seg':
                clip_feat = all_feats[seg_iter]
                text_idx = all_matches[seg_iter][-1]
                best_score = all_scores[seg_iter][-1]
                seg_iter += 1
                color_mask[mask_area] = np.concatenate([pallete[text_idx].detach().cpu().numpy(), [0.5]])
                ax.text((x1+x2)/2, (y1+y2)/2, f'{text_cands[text_idx]}', fontsize=10, color='white')


            inst_info['sem_feat'] = clip_feat
            inst_info['text_idx'] = text_idx
            seg_meta_info[inst_id+1] = inst_info

        ax.imshow(color_mask)
        ax = fig.add_subplot(122)
        ax.imshow(rgb)
        # plt.savefig(f"{result_dir}/clip_vis/{exp_name}/{f_idx}.png")
        plt.close()

        seg_dict['seg_map'] = seg_map
        seg_dict['meta_info'] = seg_meta_info

        ### save the segmentation results
        save_pkl_path = pjoin(result_dir, f'openseg_{exp_name}', 
             str(f_idx).zfill(5)+'.pkl')
        with open(save_pkl_path, "wb") as f:
            pickle.dump(seg_dict, f)

    
    
    print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 3):.4f} GB")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()