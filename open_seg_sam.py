import os, time, argparse, logging, pickle, copy
from os.path import join as pjoin
from tqdm import tqdm
import cv2 #, imageio
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.ndimage import binary_dilation
import networkx as nx

# external libs
# segmentation
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# image embedding
import clip

from transformers import AutoProcessor, AutoModel, AutoTokenizer

# self packages
from utils import *


FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

torch.cuda.set_device(2)
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

# ./rclone copy ../Openset_Panoptic_Segment/results/scene0001_00/open_set_results/ g-drive:


def cluster_with_dbscan(similarity_matrix, instance_ids, eps=0.3, min_samples=2):
    # Convert similarity to distance (1 - sim)
    distance_matrix = np.clip(1.0 - similarity_matrix, 0.0, 1.0)
    clustering = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(distance_matrix)

    # Group instances
    from collections import defaultdict
    groups = defaultdict(list)
    for inst_id, label in zip(instance_ids, labels):
        if label != -1:  # -1 means "noise"
            groups[label].append(inst_id)
    return list(groups.values())




def main():
    np.random.seed(3)

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
        'box', 'whiteboard', 'background', 
        'chair_leg', 'table_leg', 'chair_arm', 'sofa_arm', 'sofa_back', 
    ]
    pallete = get_new_pallete(len(text_cands))

    
    division_method = 'seg' # 'patch' or 'seg'
    sim_metric = 'cos'
    encoder = 'clip' # 'clip' or 'siglip'

    exp_name = f'seg_{encoder}_merge'
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
        
    # text_feats = text_feats.detach().cpu().to(torch.float32).numpy()

    # ====================================================================

    rgb_file = pjoin(rgb_path, file_names[0])
    rgb = cv2.imread(rgb_file)
    H, W, _ = rgb.shape
    yx_grid = np.mgrid[0:H, 0:W].transpose(1, 2, 0) # (H, W, 2)

    # NOTE parameters
    load_from_pkl = True
    # create segmentor
    if not load_from_pkl:
        segmentor:SAM2AutomaticMaskGenerator = create_sam_segmentor(
            points_per_side=32, min_area=(H*W) * 0.0005, 
            iou_thres=0.75, post_process=True, device=DEVICE
        )

    # collect all features for later clustering
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


        sim_func = torch.nn.CosineSimilarity(dim=-1)

        # ============================ Feature Embedding ============================
        patch_dict = {}
        img_feats = []
        inst_map = np.zeros(rgb.shape[:2], dtype=np.uint8)

        
        for inst_id, ann in enumerate(mask_dict):
            bbox_seg = ann['bbox'] # XYWH format
            x1, x2 = int(bbox_seg[0]), int(bbox_seg[0]+bbox_seg[2])
            y1, y2 = int(bbox_seg[1]), int(bbox_seg[1]+bbox_seg[3])
            
            # mask out not relevant area
            mask_area = ann['segmentation']
            valid_mask = (mask_area > 0)
            inst_map[valid_mask] = inst_id + 1
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
            
            img_feats.append(image_feat)
            # patch_dict[inst_id] = {
            #     'range': [x1, y1, x2, y2],
            #     'feat': image_feat
            # }   


        # ================= merging =================
        threshold_similar = 0.83
        num_regions = len(mask_dict)
        sim_matrix = np.zeros((num_regions, num_regions), dtype=np.float32)
        # sim_graph = nx.Graph()
        # sim_graph.add_nodes_from(range(len(mask_dict)))

        # 8-connectivity
        kernel = np.ones((3, 3), np.bool_)
        for inst_id in range(num_regions):
            sim_matrix[inst_id, inst_id] = 1.0
            mask_inst = (inst_map == (inst_id + 1))
            dilated = binary_dilation(mask_inst, structure=kernel)
            neigh_area = np.logical_and(dilated > 0, ~mask_inst)
            neighbor_ids = np.unique(inst_map[neigh_area])
            if len(neighbor_ids) == 1 and neighbor_ids[0] == 0:
                continue
            neighbor_ids = neighbor_ids[neighbor_ids > 0] # remove the background

            neigh_feats = []
            # find all similar neighboring instances
            for neigh_id in neighbor_ids:
                neigh_feats.append(img_feats[neigh_id-1])

            if len(neigh_feats) == 0:
                continue
            self_feat = img_feats[inst_id] # (D,)
            neigh_feats = torch.stack(neigh_feats, dim=0) # (M, D)
            local2neigh_sim = sim_func(self_feat, neigh_feats) # (M) 

            for i, neigh_id in enumerate(neighbor_ids):
                sim_score = local2neigh_sim[i].item()
                # if sim_score < threshold_similar:
                #     continue
                sim_matrix[inst_id, neigh_id-1] = sim_score
                sim_matrix[neigh_id-1, inst_id] = sim_score
                # sim_graph.add_edge(inst_id, neigh_id-1, weight=sim_score)
                # sim_graph.add_edge(neigh_id-1, inst_id, weight=sim_score)

        # find the connected components
        # components = list(nx.strongly_connected_components(sim_graph))
        components = cluster_with_dbscan(sim_matrix,
            instance_ids=list(range(num_regions)), 
            eps=1-threshold_similar, min_samples=1
        )

        new_inst_feat = []
        new_inst_map = np.zeros_like(inst_map, dtype=np.uint8)
        for new_inst_id, comp in enumerate(components):
            new_feat = []
            for inst_id in comp:
                # re-assign the instance id
                new_inst_map[inst_map == (inst_id + 1)] = new_inst_id + 1
                new_feat.append(img_feats[inst_id])
            new_feat = torch.mean(torch.stack(new_feat), dim=0)
            new_inst_feat.append(new_feat)


        # ========================== Assign text word ==========================
        # find the closest text word for each patch 
        # based on the cosine similarity
        all_feats = torch.stack(new_inst_feat, dim=0)
        feat_dim = all_feats.shape[-1]
        all_feats = all_feats.reshape(-1, feat_dim)

        sim_score = sim_func(all_feats.unsqueeze(1), text_feats.unsqueeze(0)).detach().cpu().numpy() # (B, T)

        # get top-3 idx and the scores
        # sort in descending order
        all_matches = np.argsort(sim_score, axis=-1)[:, -3:] # (B, 3)
        all_scores = np.take_along_axis(sim_score, all_matches, axis=1)


        # ============================ Visualization ============================
        fig = plt.figure(figsize=(20, 10))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        ax = fig.add_subplot(121)
        ax.imshow(rgb)
        sem_color_mask = np.ones((rgb.shape[0], rgb.shape[1], 4))
        sem_color_mask[:, :, 3] = 0.0
        inst_color_mask = np.ones((rgb.shape[0], rgb.shape[1], 4))
        inst_color_mask[:, :, 3] = 0.0

        seg_meta_info = {}

        # ##### Assign per segment info
        for inst_id in range(len(new_inst_feat)):
            inst_info = {}
            
            mask_area = (new_inst_map == (inst_id + 1))
            if np.count_nonzero(mask_area) < 1:
                continue
            yxs = yx_grid[mask_area] # (N, 2)
            y1, x1 = np.min(yxs, axis=0)
            y2, x2 = np.max(yxs, axis=0)


            img_feat = all_feats[inst_id].detach().cpu().to(torch.float32).numpy()
            text_idx = all_matches[inst_id][-1]
            best_score = all_scores[inst_id][-1]

            sem_color_mask[mask_area] = np.concatenate([pallete[text_idx].detach().cpu().numpy(), [0.5]])
            ax.text((x1+x2)/2, (y1+y2)/2, f'{text_cands[text_idx]}', fontsize=10, color='white')
            inst_color_mask[mask_area] = np.concatenate([np.random.random(3), [0.3]])

            contours, _ = cv2.findContours(mask_area.astype(np.uint8), 
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) \
                for contour in contours]
            cv2.drawContours(sem_color_mask, contours, -1, (0, 0, 1, 0.4), thickness=1) 

            seg_meta_info[inst_id+1] = {
                'sem_feat': img_feat
            }
        ax.imshow(sem_color_mask)


        ax = fig.add_subplot(122)
        ax.imshow(rgb)
        ax.imshow(inst_color_mask)

        plt.savefig(f"{result_dir}/clip_vis/{exp_name}/{f_idx}.png")
        plt.close()


        seg_dict = {
            'seg_map': new_inst_map, 
            'meta_info': seg_meta_info
        }
        ### save the segmentation results
        save_pkl_path = pjoin(result_dir, f'openseg_{exp_name}', 
             str(f_idx).zfill(5)+'.pkl')
        with open(save_pkl_path, "wb") as f:
            pickle.dump(seg_dict, f)

    
    
    print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 3):.4f} GB")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()