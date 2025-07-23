import os, time, argparse, logging, pickle
from tqdm import tqdm
import cv2 #, imageio
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# external libs
# segmentation
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# image embedding
import clip

# self packages
from utils import *


FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

torch.cuda.set_device(7)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    # use bfloat16
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    



def create_segmentor(
        model_name='sam2.1_hiera_large.pt', 
        cfg_name='sam2.1_hiera_l.yaml', 
        points_per_side=32):
    logging.info("Loading the segmentation model...")
    # The path is relative to the sam2 package
    model_cfg = f"configs/sam2.1/{cfg_name}"
    # This path is related to this code's location
    sam2_checkpoint = f"/home/zdeng/sam2/checkpoints/{model_name}"
    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, 
        device=DEVICE, apply_postprocessing=False
    )
    """
    Control how densely points are sampled and what the thresholds are for removing low quality or duplicate masks. 
    Generation can be automatically run on crops of the image 
    to get improved performance on smaller objects, and post-processing can remove stray pixels and holes.
    """
    # we are using a 1296 * 968 image, the items in the images is usually as samll as 20 pixel per direction. 
    # So we need 65 * 48 sampling points 
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=256,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.9,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100.0,
        use_m2m=True,
    )
    return mask_generator



# srun --time=0:5:00 -n 1 --mem-per-cpu=4g --gpus=1 --gres=gpumem:10g --pty bash
# module load stack/2024-06 && module load gcc/12.2.0 python/3.10.13 cuda/12.1.1 cudnn/8.9.7.29-12 cmake/3.27.7 eth_proxy
# source sam-env/bin/activate && cd Openset_Panoptic_Segment

# sbatch --time=0:5:00 -n 2 --mem-per-cpu=4g --gpus=1 --gres=gpumem:16g --output="logs/running.log" --wrap="python open_pano_seg.py"

def main():
    np.random.seed(3)
    print(f"using device: {DEVICE}")

    seq_name = 'scene0001_00'
    # data_dir = '/scratch/zdeng/datasets/scannet'
    data_dir = '/cluster/scratch/dengzi/scannet'
    result_dir = os.path.join('results', seq_name)
    os.makedirs(os.path.join(result_dir, 'anns'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'open_seg_results'), exist_ok=True)

    # image is 1296 * 968
    rgb_path = os.path.join(data_dir, seq_name, 'color')

    file_names = os.listdir(rgb_path)
    # strip the extensions and sort by the numeric part of the file names
    file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))
    num_frame = len(file_names)
    step = 5
    num_frame = 100 * step

    clip_model, prep_clip = create_feature_extractor(device=DEVICE)
    # clip_prompt_cands = [
    #     'cabinet', 'bed', 'chair', 'truck', 'sofa', 'table', 'door', 
    #     'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 
    #     'pillow', 'refridgerator', 'television', 'shower curtain', 
    #     'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 
    #     'bag',  'otherfurniture', 
    #     'wall', 'floor', 'blinds', 'shelves', 'dresser', 'mirror', 
    #     'floor mat', 'clothes', 'ceiling', 'books', 'paper', 'towel', 
    #     'box', 'whiteboard', 'otherstructure', 'otherprop'
    # ]
    # text_cands = clip.tokenize(clip_prompt_cands).to(DEVICE)
    # with torch.no_grad():
    #     text_features = clip_model.encode_text(text_cands)

    num_sam_samples = 32
    # clip_pca_dim = 128

    # masks = []
    load_from_pkl = False
    if not load_from_pkl:
        segmentor:SAM2AutomaticMaskGenerator = create_segmentor(
            points_per_side=num_sam_samples
        )
    
    for idx in tqdm(range(0, num_frame, step)):
        logging.info(f"Processing frame: {file_names[idx]}")
        
        rgb_file = os.path.join(rgb_path, file_names[idx])
        rgb = cv2.imread(rgb_file)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # ========================= SAM =========================
        if load_from_pkl:
            with open(f"{result_dir}/mask_dict.pkl", "rb") as f:
                mask_dict = pickle.load(f)
        else:
            with torch.no_grad():
                mask_dict = segmentor.generate(rgb)
            if len(mask_dict) == 0:
                logging.warning("No mask is segmented by sam2, skipping..")
                continue

            mask_dict = sorted(mask_dict, key=(lambda x: x['area']), reverse=True)

            # store it into files
            with open(f"{result_dir}/{idx}_mask_{num_sam_samples}samples.pkl", "wb") as f:
                pickle.dump(mask_dict, f)
            show_anns(mask_dict, rgb, f"{result_dir}/anns/{idx}_{num_sam_samples}samples.png")

        logging.info(f"Total candidates: {len(mask_dict)}")
        # =======================================================
        

        # fig = plt.figure(figsize=(20, 10))
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # ax = fig.add_subplot(121)
        # ax.imshow(rgb)
        # color_mask = np.ones((rgb.shape[0], rgb.shape[1], 4))
        # color_mask[:, :, 3] = 0 # set all color mask to 0 alpha

        # ========================= CLIP =========================
        # method 1 global context
        # with torch.no_grad():
        #     image_feat_global = clip_model.encode_image(
        #         prep_clip(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE))
        # # print("CLIP feat dim: ", image_feat_global.shape)
        # image_feat_global = image_feat_global.detach().cpu().to(torch.float32).numpy()


        # method 2 patch context
        all_feats = []
        # cut the rgb into N*M equal parts and get the range and the cores patch as a dict
        patch_num = (8, 6) # num_W, num_H
        patch_size = (rgb.shape[1] // patch_num[0], rgb.shape[0] // patch_num[1])
        patch_dict = {}
        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                x1, x2 = i * patch_size[0], (i+1) * patch_size[0]
                y1, y2 = j * patch_size[1], (j+1) * patch_size[1]
                with torch.no_grad():
                    image_feat_patch = clip_model.encode_image(
                        prep_clip(Image.fromarray(rgb[y1:y2, x1:x2])).unsqueeze(0).to(DEVICE))
                image_feat_patch = image_feat_patch.detach().cpu().to(torch.float32).numpy()

                patch_dict[(i, j)] = {
                    'range': [x1, y1, x2, y2],
                    'feat': image_feat_patch
                }

        # =======================================================

        seg_dict = {}
        seg_meta_info = {}
        seg_map = np.zeros(rgb.shape[:2], dtype=np.uint8)

        for inst_id, ann in enumerate(mask_dict):
            inst_info = {}
            bbox_seg = ann['bbox'] # XYWH format
            x1, x2 = int(bbox_seg[0]), int(bbox_seg[0]+bbox_seg[2])
            y1, y2 = int(bbox_seg[1]), int(bbox_seg[1]+bbox_seg[3])
            inst_info['bbox'] = [x1, y1, x2, y2]
            # cropped_rgb = rgb[y1:y2, x1:x2]

            mask_area = ann['segmentation']
            inst_info['area'] = np.count_nonzero(mask_area)
            seg_map[mask_area] = inst_id + 1
            # color_mask[mask_area] = np.concatenate([np.random.random(3), [0.5]])

            inst_info['score_iou'] = ann['predicted_iou']

            # for the global context, just assign to all segs
            # inst_info['sem_feat'] = image_feat_global

            # for the patch context, find the patch with largest overlap
            max_overlap = 0
            clip_feat = None
            for key, patch_info in patch_dict.items():
                x1, y1, x2, y2 = patch_info['range']
                overlap = np.count_nonzero(mask_area[y1:y2, x1:x2])
                if overlap > max_overlap:
                    max_overlap = overlap
                    clip_feat = patch_info['feat']
            inst_info['sem_feat'] = clip_feat


            # cropped_rgb = prep_clip(Image.fromarray(cropped_rgb)).unsqueeze(0).to(DEVICE)
            # with torch.no_grad():
            #     # original size is 512
            #     # image_feat = clip_model.encode_image(cropped_rgb)
            #     logits_per_image, logits_per_text = clip_model(cropped_rgb, text_cands)
            #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            #     class_id = np.argmax(probs)
            
            # print(image_feat.shape)
            # seg_dict["clip_feat"] = image_feat

            # bbox = ann['bbox']
            # ax.text(bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, f'{clip_prompt_cands[class_id]}')

            seg_meta_info[inst_id+1] = inst_info

        # ax.imshow(color_mask)
        # ax = fig.add_subplot(122)
        # ax.imshow(rgb)
        # plt.savefig(f"{result_dir}/anns/{idx}_clip.png")
        # plt.close()

        seg_dict['seg_map'] = seg_map
        seg_dict['meta_info'] = seg_meta_info
        save_pkl_path = os.path.join(result_dir, 'open_seg_results', str(idx).zfill(5)+'.pkl')
        with open(save_pkl_path, "wb") as f:
            pickle.dump(seg_dict, f)

        torch.cuda.empty_cache()

    print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 3):.4f} GB")


if __name__ == '__main__':
    main()