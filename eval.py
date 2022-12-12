# %%
import argparse
from tqdm import tqdm

import os
import random
import numpy as np
import pandas as pd

import torch
import clip

from data import *
from model import *

# %%
def main(config):
    BATCH_SIZE = 128
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    data_df = get_data(preprocess, config.data_path, config.hard_mode, config.multi_mode)

    train, val, test = split_data(data_df)
    test_data = headcam_data(test, multi_mode=config.multi_mode)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # %%
    HARD_STR = "_hard" if config.hard_mode else ""
    MULTI_STR = "_multi" if config.multi_mode else ""
    models = sorted([f[:-3] for f in os.listdir("models") if f.endswith(f"{HARD_STR}{MULTI_STR}.pt")])
    model_configs = [m.split("_") for m in models]
    model_configs = [{f:v for f,v in zip(config[::2], config[1::2])} for config in model_configs]
    model_configs = pd.DataFrame(model_configs, dtype="float")
    model_configs['path'] = models

    # %%
    all_img_f1s = []
    all_txt_f1s = []
    all_mat_f1s = []

    indiv_img_f1s = []
    indiv_txt_f1s = []

    for img_lambda, txt_lambda, path in tqdm(zip(model_configs["img"], model_configs["txt"], model_configs["path"])):
        # f'models/img_{config.img_lambda}_txt_{config.txt_lambda}_hid_{config.hidden_dim}_lr_{config.learning_rate}.pt'
        mlp = torch.load(f'models/{path}.pt')

        all_preds = []
        all_labels = []

        for test_batch in test_loader:
            preds, _, _ = train_step(test_batch, mlp, None, img_lambda, txt_lambda, eval=True, multi_mode=config.multi_mode)
            _, labels = test_batch
            all_preds.append(preds)
            all_labels.append(labels)

        img_preds, txt_preds, mat_preds = zip(*all_preds)
        img_labels, txt_labels, mat_labels = zip(*all_labels)
        img_preds = torch.cat(img_preds, dim=0)
        txt_preds = torch.cat(txt_preds, dim=0)
        mat_preds = torch.cat(mat_preds, dim=0)
        img_labels = torch.cat(img_labels, dim=0)
        txt_labels = torch.cat(txt_labels, dim=0)
        mat_labels = torch.cat(mat_labels, dim=0)
        preds = (img_preds, txt_preds, mat_preds)
        labels = (img_labels, txt_labels, mat_labels)

        img_f1, txt_f1, mat_f1 = macro_f1(preds, labels, config.multi_mode)
        all_img_f1s.append(img_f1)
        all_txt_f1s.append(txt_f1)
        all_mat_f1s.append(mat_f1)

        img_f1_indiv, txt_f1_indiv = indiv_f1(preds, labels, config.multi_mode)
        indiv_img_f1s.append(img_f1_indiv)
        indiv_txt_f1s.append(txt_f1_indiv)
        
    # %%
    results = model_configs
    results['img_f1'] = all_img_f1s
    results['txt_f1'] = all_txt_f1s
    results['mat_f1'] = all_mat_f1s
    results['indiv_img_f1s'] = indiv_img_f1s
    results['indiv_txt_f1s'] = indiv_txt_f1s

    results.to_csv(config.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="frame-extraction/all_namings_cleaned.csv")
    parser.add_argument("--hard_mode", action='store_true')
    parser.add_argument("--multi_mode", action='store_true')
    parser.add_argument("--out_path", type=str, default="eval.csv")
    main(parser.parse_args())