import os, sys, random, requests

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download

sys.path.append(
    os.path.join(os.getcwd(), "../Grounded-Segment-Anything/GroundingDINO")
)

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, predict

# segment anything
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# CLIPSeg
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def load_model_hf(repo_id, filename, ckpt_config_filename, device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    model.to(device)
    return model


def download_image(url):
    return Image.open(requests.get(url, stream=True).raw)


def load_image_for_dino(image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dino_image, _ = transform(image, None)
    return dino_image


def dino_detection(
    model,
    image,
    image_array,
    category_names,
    category_name_to_id,
    box_threshold,
    text_threshold,
    device,
    visualize=False,
):
    detection_prompt = " . ".join(category_names)
    dino_image = load_image_for_dino(image)
    dino_image = dino_image.to(device)
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=dino_image,
            caption=detection_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            remove_combined=True,
        )
    category_ids = [category_name_to_id[phrase] for phrase in phrases]

    if visualize:
        annotated_frame = annotate(
            image_source=image_array, boxes=boxes, logits=logits, phrases=phrases
        )
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
        visualization = Image.fromarray(annotated_frame)
        return boxes, category_ids, visualization
    else:
        return boxes, category_ids, phrases


def sam_masks_from_dino_boxes(predictor, image_array, boxes, device):
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_array.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_array.shape[:2]
    ).to(device)
    thing_masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return thing_masks


def preds_to_semantic_inds(preds, threshold):
    flat_preds = preds.reshape((preds.shape[0], -1))
    # Initialize a dummy "unlabeled" mask with the threshold
    flat_preds_with_treshold = torch.full(
        (preds.shape[0] + 1, flat_preds.shape[-1]), threshold
    )
    flat_preds_with_treshold[1 : preds.shape[0] + 1, :] = flat_preds

    # Get the top mask index for each pixel
    semantic_inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape(
        (preds.shape[-2], preds.shape[-1])
    )

    return semantic_inds


def clipseg_segmentation(
    processor, model, image, category_names, background_threshold, device
):
    inputs = processor(
        text=category_names,
        images=[image] * len(category_names),
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    if len(logits.shape) == 2:
      logits = logits.unsqueeze(0)
    # resize the outputs
    upscaled_logits = nn.functional.interpolate(
        logits.unsqueeze(1),
        size=(image.size[1], image.size[0]),
        mode="bilinear",
    )
    preds = torch.sigmoid(upscaled_logits.squeeze(dim=1))
    semantic_inds = preds_to_semantic_inds(preds, background_threshold)
    return preds, semantic_inds


def semantic_inds_to_shrunken_bool_masks(
    semantic_inds, shrink_kernel_size, num_categories
):
    shrink_kernel = np.ones((shrink_kernel_size, shrink_kernel_size))

    bool_masks = torch.zeros((num_categories, *semantic_inds.shape), dtype=bool)
    for category in range(num_categories):
        binary_mask = semantic_inds == category
        shrunken_binary_mask_array = (
            ndimage.binary_erosion(binary_mask.numpy(), structure=shrink_kernel)
            if shrink_kernel_size > 0
            else binary_mask.numpy()
        )
        bool_masks[category] = torch.from_numpy(shrunken_binary_mask_array)

    return bool_masks


def clip_and_shrink_preds(semantic_inds, preds, shrink_kernel_size, num_categories):
    # convert semantic_inds to shrunken bool masks
    bool_masks = semantic_inds_to_shrunken_bool_masks(
        semantic_inds, shrink_kernel_size, num_categories
    ).to(preds.device)

    sizes = [
        torch.sum(bool_masks[i].int()).item() for i in range(1, bool_masks.size(0))
    ]
    max_size = max(sizes)
    relative_sizes = [size / max_size for size in sizes] if max_size > 0 else sizes

    # use bool masks to clip preds
    clipped_preds = torch.zeros_like(preds)
    for i in range(1, bool_masks.size(0)):
        float_mask = bool_masks[i].float()
        clipped_preds[i - 1] = preds[i - 1] * float_mask

    return clipped_preds, relative_sizes


def sample_points_based_on_preds(preds, N):
    height, width = preds.shape
    weights = preds.ravel()
    indices = np.arange(height * width)

    # Randomly sample N indices based on the weights
    sampled_indices = random.choices(indices, weights=weights, k=N)

    # Convert the sampled indices into (col, row) coordinates
    sampled_points = [(index % width, index // width) for index in sampled_indices]

    return sampled_points


def upsample_pred(pred, image_source):
    pred = pred.unsqueeze(dim=0)
    original_height = image_source.shape[0]
    original_width = image_source.shape[1]

    larger_dim = max(original_height, original_width)
    aspect_ratio = original_height / original_width

    # upsample the tensor to the larger dimension
    upsampled_tensor = F.interpolate(
        pred, size=(larger_dim, larger_dim), mode="bilinear", align_corners=False
    )

    # remove the padding (at the end) to get the original image resolution
    if original_height > original_width:
        target_width = int(upsampled_tensor.shape[3] * aspect_ratio)
        upsampled_tensor = upsampled_tensor[:, :, :, :target_width]
    else:
        target_height = int(upsampled_tensor.shape[2] * aspect_ratio)
        upsampled_tensor = upsampled_tensor[:, :, :target_height, :]
    return upsampled_tensor.squeeze(dim=1)


def sam_mask_from_points(predictor, image_array, points):
    points_array = np.array(points)
    # we only sample positive points, so labels are all 1
    points_labels = np.ones(len(points))
    # we don't use predict_torch here cause it didn't seem to work...
    masks, scores, logits = predictor.predict(
        point_coords=points_array,
        point_labels=points_labels,
    )
    # max over the 3 segmentation levels
    total_pred = torch.max(torch.sigmoid(torch.tensor(logits)), dim=0)[0].unsqueeze(
        dim=0
    )
    # logits are 256x256 -> upsample back to image shape
    upsampled_pred = upsample_pred(total_pred, image_array)
    return upsampled_pred

def generate_panoptic_mask(
    image,
    thing_category_names,
    stuff_category_names,
    category_name_to_id,
    dino_model,
    sam_predictor,
    clipseg_processor,
    clipseg_model,
    device,
    dino_box_threshold=0.3,
    dino_text_threshold=0.25,
    segmentation_background_threshold=0.1,
    shrink_kernel_size=20,
    num_samples_factor=1000,
):
    image = image.convert("RGB")
    image_array = np.asarray(image)

    # compute SAM image embedding
    sam_predictor.set_image(image_array)

    # detect boxes for "thing" categories using Grounding DINO
    thing_category_ids = []
    thing_masks = []
    thing_boxes = []
    if len(thing_category_names) > 0:
        thing_boxes, thing_category_ids, _ = dino_detection(
            dino_model,
            image,
            image_array,
            thing_category_names,
            category_name_to_id,
            dino_box_threshold,
            dino_text_threshold,
            device,
        )
        if len(thing_boxes) > 0:
            # get segmentation masks for the thing boxes
            thing_masks = sam_masks_from_dino_boxes(
                sam_predictor, image_array, thing_boxes, device
            )

    if len(stuff_category_names) > 0:
        # get rough segmentation masks for "stuff" categories using CLIPSeg
        clipseg_preds, clipseg_semantic_inds = clipseg_segmentation(
            clipseg_processor,
            clipseg_model,
            image,
            stuff_category_names,
            segmentation_background_threshold,
            device,
        )
        # remove things from stuff masks
        clipseg_semantic_inds_without_things = clipseg_semantic_inds.clone()
        if len(thing_boxes) > 0:
            combined_things_mask = torch.any(thing_masks, dim=0)
            clipseg_semantic_inds_without_things[combined_things_mask[0]] = 0
        # clip CLIPSeg preds based on non-overlapping semantic segmentation inds (+ optionally shrink the mask of each category)
        # also returns the relative size of each category
        clipsed_clipped_preds, relative_sizes = clip_and_shrink_preds(
            clipseg_semantic_inds_without_things,
            clipseg_preds,
            shrink_kernel_size,
            len(stuff_category_names) + 1,
        )
        # get finer segmentation masks for the "stuff" categories using SAM
        sam_preds = torch.zeros_like(clipsed_clipped_preds)
        for i in range(clipsed_clipped_preds.shape[0]):
            clipseg_pred = clipsed_clipped_preds[i]
            # for each "stuff" category, sample points in the rough segmentation mask
            num_samples = int(relative_sizes[i] * num_samples_factor)
            if num_samples == 0:
                continue
            points = sample_points_based_on_preds(
                clipseg_pred.cpu().numpy(), num_samples
            )
            if len(points) == 0:
                continue
            # use SAM to get mask for points
            pred = sam_mask_from_points(sam_predictor, image_array, points)
            sam_preds[i] = pred
        sam_semantic_inds = preds_to_semantic_inds(
            sam_preds, segmentation_background_threshold
        )

    # combine the thing inds and the stuff inds into panoptic inds
    panoptic_inds = (
        sam_semantic_inds.clone()
        if len(stuff_category_names) > 0
        else torch.zeros(image_array.shape[0], image_array.shape[1], dtype=torch.long)
    )
    ind = len(stuff_category_names) + 1
    for thing_mask in thing_masks:
        # overlay thing mask on panoptic inds
        panoptic_inds[thing_mask.squeeze(dim=0)] = ind
        ind += 1

    return panoptic_inds, thing_category_ids



def main():

    seq_name = 'scene0001_00'
    data_dir = '/scratch/zdeng/datasets/scannet'
    rgb_path = os.path.join(data_dir, seq_name, 'color')

    file_names = os.listdir(rgb_path)
    # strip the extensions and sort by the numeric part of the file names
    file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))
    rgb_file = os.path.join(rgb_path, file_names[0])
    image = cv2.imread(rgb_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    thing_category_names = [
        'cabinet', 'bed', 'chair', 'truck', 'sofa', 'table', 'door', 
        'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 
        'pillow', 'refridgerator', 'television', 'shower curtain', 
        'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 
        'bag',  'otherfurniture']
    stuff_category_names = [
        'wall', 'floor', 'blinds', 'shelves', 'dresser', 'mirror', 
        'floor mat', 'clothes', 'ceiling', 'books', 'paper', 'towel', 
        'box', 'whiteboard', 'otherstructure', 'otherprop']


    category_names = thing_category_names + stuff_category_names
    category_name_to_id = {
        category_name: i for i, category_name in enumerate(category_names)
    }


    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = load_model_hf(
        ckpt_repo_id, ckpt_filename, ckpt_config_filename, device
    )


    model_name='sam2.1_hiera_large.pt'
    cfg_name='sam2.1_hiera_l.yaml'
    # The path is relative to the sam2 package
    model_cfg = f"configs/sam2.1/{cfg_name}"
    # This path is related to this code's location
    sam2_checkpoint = f"/home/zdeng/sam2/checkpoints/{model_name}"
    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, 
        device=device, apply_postprocessing=False
    )
    sam_predictor = SAM2ImagePredictor(sam2)


    clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    )
    clipseg_model.to(device)

    panoptic_inds, thing_category_ids = generate_panoptic_mask(
        image,
        thing_category_names,
        stuff_category_names,
        category_name_to_id,
        groundingdino_model,
        sam_predictor,
        clipseg_processor,
        clipseg_model,
        device,
    )