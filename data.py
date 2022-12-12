from PIL import Image
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

OBJS = ['ball', 'brush', 'car', 'manu', 'tima', 'zem']
OBJS_HARD = ['ball', 'wug', 'brush', 'car', 'manu', 'tima', 'zem', 'blicket']

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# Data retrieval

def get_image(row, preprocess_fn):
    time = str(row["time"]).rstrip('0').rstrip('.')
    img_label = row["img_label"]
    img_label = "ball" if img_label == "wug" else "zem" if img_label == "blicket" else img_label
    img_suffix = "".join([f"_{img_lbl}" for img_lbl in OBJS if row[f"img_{img_lbl}"] == 1]) if img_label == "multi" else ""
    img_path = f'frame-extraction/frames/{row["child_id"]}/{img_label}/{time}{img_suffix}.jpg'
    img = Image.open(img_path)
    return preprocess_fn(img)

def get_data(preprocess_fn, path="frame-extraction/all_namings_cleaned.csv", hard_mode=False, multi_mode=False):
    stoi = {obj: idx for idx, obj in enumerate(OBJS_HARD if hard_mode else OBJS)}

    data_df = pd.read_csv(path)
    if not multi_mode:
        data_df['mat_label'] = data_df.apply(lambda row: row['txt_label'] == row['img_label'], axis = 1)
        data_df['txt_label_int'] = [stoi[txt] for txt in data_df['txt_label']] 
        data_df['img_label_int'] = [stoi[img] for img in data_df['img_label']]
    data_df['img'] = data_df.apply(lambda row: get_image(row, preprocess_fn), axis=1)

    return data_df

def split_data(data_df, val_prop=.2, test_prop=.2):
    return np.split(data_df.sample(frac=1), [int((1-test_prop-val_prop)*len(data_df)), int((1-test_prop)*len(data_df))])

# Data loading

class headcam_data(torch.utils.data.Dataset):
    def __init__(self, source, multi_mode=False):
        self.len = len(source)
        self.data = source
        self.multi_mode = multi_mode

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = row['img']
        img_label = np.array(row["img_ball":"img_zem"], dtype=float) if self.multi_mode else row["img_label_int"]
        txt = row["utterance"]
        txt_label = np.array(row["txt_ball":"txt_zem"], dtype=float) if self.multi_mode else row["txt_label_int"]
        mat_label = row["mat_label"]
        return (img, txt), (img_label, txt_label, mat_label)

    def __len__(self):
        return self.len

# Visualisation

def show_batch(dl):
    for batch in dl:
        (images, _), _ = batch
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20],nrow=5).permute(1,2,0))
        break