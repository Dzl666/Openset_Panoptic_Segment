import os
from os.path import join as pjoin
from gradio_client import Client, handle_file

seq_name = 'scene0000_00'
data_dir = '/scratch/zdeng/datasets/scannet'

# image is 1296 * 968
rgb_path = pjoin(data_dir, seq_name, 'color')
file_names = os.listdir(rgb_path)
file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

img_path = pjoin(rgb_path, file_names[0])


client = Client("https://stevengrove-yolo-world.hf.space/--replicas/vp0az/")
result = client.predict(
    handle_file(img_path), 
    "bedroom, bureau, chair, computer, computer chair, computer desk, curtain, table, drawer, electronic, floor, office, office chair, office desk, office supply, room, speaker, stool, swivel chair", 
    50, 
    0.25,	# 'Score Threshold'
    0.5,	# NMS Threshold
    api_name="/partial"
)
# print(result)
