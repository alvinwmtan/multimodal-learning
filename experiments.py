# %%
import argparse
from tqdm import tqdm

import random
import numpy as np
import pandas as pd
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

# %%
from sklearn.metrics import f1_score

# %%
from model import *
from data import *

# %% [markdown]
# ## Setup

# %%
def main(config):
    FEAT_SIZE = 512

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # %%
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # %%
    HARD_STR = "_hard" if config.hard_mode else ""
    MULTI_STR = "_multi" if config.multi_mode else ""
    writer = SummaryWriter(f"runs/img_{config.img_lambda}_txt_{config.txt_lambda}_hid_{config.hidden_dim}_lr_{config.learning_rate}{MULTI_STR}{HARD_STR}")

    # %% [markdown]
    # ## Data

    # %%
    data_df = get_data(preprocess, config.data_path, config.hard_mode, config.multi_mode)

    # %%
    train, val, test = split_data(data_df)

    # %%
    train_data = headcam_data(train, multi_mode=config.multi_mode)
    val_data = headcam_data(val, multi_mode=config.multi_mode)
    test_data = headcam_data(test, multi_mode=config.multi_mode)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=True)

    # %% [markdown]
    # ## Model

    # %%
    mlp = MultiTaskMLP(FEAT_SIZE*2, config.hidden_dim, config.num_classes, config.num_classes, model)
    optim = torch.optim.Adam(mlp.parameters(), lr=config.learning_rate)

    # %%
    n_batches = 0
    best_val_loss = 100

    for epoch in tqdm(range(config.num_epochs)):
        # Train
        for batch in train_loader:
            preds, loss, indiv_losses = train_step(batch, mlp, optim, config.img_lambda, config.txt_lambda, multi_mode=config.multi_mode)
            img_loss, txt_loss, mat_loss = indiv_losses
            writer.add_scalar("Loss/train", loss, n_batches)
            writer.add_scalar("Image loss/train", img_loss, n_batches)
            writer.add_scalar("Text loss/train", txt_loss, n_batches)
            writer.add_scalar("Match loss/train", mat_loss, n_batches)
            n_batches += 1
        
        # Validate
        val_batch = next(cycle(iter(val_loader)))
        preds, loss, indiv_losses = train_step(val_batch, mlp, optim, config.img_lambda, config.txt_lambda, eval=True, multi_mode=config.multi_mode)
        writer.add_scalar("Loss/val", loss, n_batches)
        writer.add_scalar("Image loss/val", img_loss, n_batches)
        writer.add_scalar("Text loss/val", txt_loss, n_batches)
        writer.add_scalar("Match loss/val", mat_loss, n_batches)
        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(mlp, f'models/img_{config.img_lambda}_txt_{config.txt_lambda}_hid_{config.hidden_dim}_lr_{config.learning_rate}{MULTI_STR}{HARD_STR}.pt')

        _, labels = val_batch
        img_f1, txt_f1, mat_f1 = macro_f1(preds, labels, config.multi_mode)
        writer.add_scalar("Image F1 score/val", img_f1, n_batches)
        writer.add_scalar("Text F1 score/val", txt_f1, n_batches)
        writer.add_scalar("Match F1 score/val", mat_f1, n_batches)

    # torch.save(mlp, f'models/img_{config.img_lambda}_txt_{config.txt_lambda}_hid_{config.hidden_dim}_lr_{config.learning_rate}{HARD_STR}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--img_lambda", type=float, default=0.33)
    parser.add_argument("--txt_lambda", type=float, default=0.33)
    parser.add_argument("--data_path", type=str, default="frame-extraction/all_namings_cleaned.csv")
    parser.add_argument("--hard_mode", action='store_true')
    parser.add_argument("--multi_mode", action='store_true')
    main(parser.parse_args())