import os, time, argparse, logging, pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

import clip

# self
from utils import *

FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

torch.cuda.set_device(7)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ./rclone copy ../Openset_Panoptic_Segment/results/text_embed/clip_text_embed.pkl g-drive:

def main():

    result_dir = os.path.join('results', 'text_embed')
    os.makedirs(result_dir, exist_ok=True)

    clip_model, prep_clip = create_feature_extractor(device=DEVICE)

    # ===================== NYU 40 ========================
    # clip_text_cands = [
    #     'cabinet', 'bed', 'chair', 'truck', 'sofa', 'table', 'door', 
    #     'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 
    #     'pillow', 'refridgerator', 'television', 'shower curtain', 
    #     'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 
    #     'bag',  'otherfurniture', 
    #     'wall', 'floor', 'blinds', 'shelves', 'dresser', 'mirror', 
    #     'floor mat', 'clothes', 'ceiling', 'books', 'paper', 'towel', 
    #     'box', 'whiteboard', 'otherstructure', 'otherprop'
    # ]

    # ======= Subset of NYU 40 + some subdivided categories ========
    clip_text_cands = [
        'cabinet', 'bed', 'chair', 'truck', 'sofa', 'table', 'door', 
        'window', 'bookshelf', 'picture', 'desk', 'curtain', 
        'pillow', 'nightstand', 'toilet', 'sink', 'lamp'
        'wall', 'floor', 'blinds', 'shelves', 'dresser', 'mirror', 
        'floor mat', 'clothes', 'ceiling', 'books', 'paper', 'towel', 
        'box', 'whiteboard',
        'chair leg', 'table leg', 'chair arm', 'sofa arm', 'sofa back', 
        'chairleg', 'tableleg', 'chairarm', 'sofaarm', 'sofaback', 
        'chair_leg', 'table_leg', 'chair_arm', 'sofa_arm', 'sofa_back', 
    ]

    text_cands = clip.tokenize(clip_text_cands).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_cands)

    text_features = text_features.detach().cpu().to(torch.float32).numpy()

    text_embed = {}
    for i in range(text_features.shape[0]):
        text_embed[clip_text_cands[i]] = text_features[i]

    # Save the text features
    save_pkl_path = os.path.join(result_dir, 'clip_text_embed.pkl')
    with open(save_pkl_path, 'wb') as f:
        pickle.dump(text_embed, f)

if __name__ == '__main__':
    main()