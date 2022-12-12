# %%
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import clip

from data import *
from model import *

# %%
BATCH_SIZE = 1
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

reps_df = pd.read_csv("reps.csv")
reps_df['img'] = reps_df.apply(lambda row: preprocess(Image.open(row['img_path'])), axis=1)

reps_data = headcam_data(reps_df)
reps_loader = torch.utils.data.DataLoader(reps_data, batch_size=BATCH_SIZE, shuffle=False)

# %%
models = [f[:-3] for f in os.listdir("models") if f.endswith(".pt")]
model_configs = [m.split("_") for m in models]
model_configs = [{f:v for f,v in zip(config[::2], config[1::2])} for config in model_configs]
model_configs = pd.DataFrame(model_configs, dtype="float")
model_configs['path'] = models

# %%
def get_activation():
    def hook(model, input, output):
        rep_tensors.append(output.detach().numpy())
    return hook

# %%
all_closeness = []

for img_lambda, txt_lambda, path in tqdm(zip(model_configs["img"], model_configs["txt"], model_configs["path"])):
    mlp = torch.load(f'models/{path}.pt')

    # get representations
    rep_tensors = []
    h = mlp.layer3.register_forward_hook(get_activation())
    for rep in reps_loader:
        input, labels = rep
        out = mlp(input)
    h.remove()
    
    # get correlations
    rep_arr = [[np.corrcoef(rep_tensors[img], rep_tensors[txt+6])[0,1] for txt in range(6)] for img in range(6)]
    rep_arr = np.array(rep_arr)

    # get closeness metric
    closeness = (np.trace(rep_arr) / 6) - ((np.sum(rep_arr) - np.trace(rep_arr)) / (6*6 - 6))
    all_closeness.append(closeness)

    # make plot
    fig, ax = plt.subplots()
    im = ax.imshow(rep_arr, cmap="YlGn")
    ax.set_xticks(np.arange(len(OBJS)), labels=OBJS)
    ax.set_yticks(np.arange(len(OBJS)), labels=OBJS)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", \
        rotation_mode="anchor")
    for img in range(len(OBJS)):
        for txt in range(len(OBJS)):
            text = ax.text(txt, img, f'{rep_arr[img, txt]:.2f}', \
                ha="center", va="center", color=("k", "w")[int(im.norm(rep_arr[img, txt]) > (rep_arr.max()/2))])
    plt.xlabel("Text")
    plt.ylabel("Image")
    plt.title(f"$\lambda_{{img}}$ = {img_lambda}, $\lambda_{{txt}}$ = {txt_lambda}")
    fig.tight_layout()
    plt.savefig(f"corr_plots/img_{img_lambda}_txt_{txt_lambda}.png")
    plt.close()
    pass
    
# %%
results = model_configs
results['closeness'] = all_closeness
results.to_csv("closeness.csv")
