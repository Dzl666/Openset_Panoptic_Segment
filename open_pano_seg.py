import os, time, argparse, pickle
from tqdm import tqdm
import cv2 #, imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    # use bfloat16
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def show_anns(anns, rgb, figure_name='mask_map', borders=False):
    if len(anns) == 0:
        return
    anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    color_mask = np.ones((rgb.shape[0], rgb.shape[1], 4))
    color_mask[:, :, 3] = 0 # set all color mask to 0 alpha
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(rgb)

    for ann in anns:
        mask_area = ann['segmentation']
        color_mask[mask_area] = np.concatenate([np.random.random(3), [0.7]])

        sample_pt = ann['point_coords']
        score = ann['predicted_iou']

        ax.text(sample_pt[0][0], sample_pt[0][1], f'{score:.3f}')

        if borders:
            contours, _ = cv2.findContours(mask_area.astype(np.uint8), 
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) \
                for contour in contours]
            cv2.drawContours(color_mask, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(color_mask)

    plt.savefig(f"{figure_name}.png")
    plt.close()
    return
    
# srun --time=0:10:00 -n 1 --mem-per-cpu=4g --gpus=1 --gres=gpumem:8g --pty bash
# module load stack/2024-06 && module load gcc/12.2.0 python/3.10.13 cuda/12.1.1 cudnn/8.9.7.29-12 cmake/3.27.7 eth_proxy
# source sam-env/bin/activate && cd Openset_Panoptic_Segment

def main():
    np.random.seed(3)
    print(f"using device: {device}")

    seq_name = 'scene0000_00'
    data_dir = '/cluster/scratch/dengzi/scannet'

    rgb_path = os.path.join(data_dir, seq_name, 'color')

    # file_names = sorted(os.listdir(rgb_path))
    file_names = os.listdir(rgb_path)
    num_frame = len(file_names)

    # The path is relative to the sam2 package
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # This path is related to this code's location
    sam2_checkpoint = "/cluster/home/dengzi/sam2/checkpoints/sam2.1_hiera_large.pt"

    print("Loading the model...")
    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, 
        device=device, apply_postprocessing=False
    )
    """
    There are several tunable parameters in automatic mask generation that 
    control how densely points are sampled and what the thresholds are for 
    removing low quality or duplicate masks. 
    Additionally, generation can be automatically run on crops of the image 
    to get improved performance on smaller objects, and post-processing can 
    remove stray pixels and holes.
    """
    # we are using a 1296 * 968 image, the items in the images is usually as samll as 20 pixel per direction. 
    # So we need 65 * 48 sampling points 
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=128,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.88,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100.0,
        use_m2m=True,
    )

    # masks = []
    load_from_pkl = False
    
    for idx in tqdm(range(1)):
        
        rgb_file = os.path.join(rgb_path, file_names[idx])
        rgb = cv2.imread(rgb_file)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if not load_from_pkl:
            with torch.no_grad():
                mask_dict = mask_generator.generate(rgb)
            with open("mask_dict.pkl", "wb") as f:
                pickle.dump(mask_dict, f)
        else:
            with open("mask_dict.pkl", "rb") as f:
                mask_dict = pickle.load(f)

        print("Total candidates: ", len(mask_dict))
        show_anns(mask_dict, rgb, f'mask_{idx}')

        print(mask_dict[0]['bbox'])
        print(mask_dict[0]['point_coords'])
        print(mask_dict[0]['crop_box'])

    print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 3):.4f} GB")




if __name__ == '__main__':
    main()