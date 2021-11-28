import os
import random
from functools import partial
from glob import glob
from os.path import join, basename
import pickle
import os
import torch
import copy
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from imageio import mimread
from sklearn.metrics import f1_score as measure_f1_score
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from itertools import combinations
import copy
from stylegan_infer import Model

from config import opt


measure_f1_score = partial(measure_f1_score, average='micro')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

writer = SummaryWriter(log_dir=os.path.join(opt.checkpoint_dir, 'tf_log'))


class Dataloader:
    def __init__(self):
        self.model = Model(path_to_checkpoint=join(opt.path_to_stylegan_checkpoints, 'neutral'), device=opt.device)

    def get_batch(self, batch_size):
        noise = torch.randn(1, 512, device=opt.device)
        images = []
        emotions = []
        for b in range(batch_size):
            emo = random.choice(opt.emotion_list)
            emotions.append(emo)
            self.model.load_checkpoint(join(opt.path_to_stylegan_checkpoints, emo))
            images.append(self.model.inference(1, noise)[2])
        comb = list(combinations(range(batch_size), 2))
        random.shuffle(comb)
        inputs_b, targets_b, emotions_b = [], [], []
        for b in range(batch_size):
            inputs_b.append(images[comb[b][0]])
            targets_b.append(images[comb[b][1]])
            emotions_b.append((emotions[comb[b][0]], emotions[comb[b][1]]))
        return torch.cat(inputs_b).add(1).div(2), torch.cat(targets_b).add(1).div(2), emotions_b



if __name__ == "__main__":
    dataloader = Dataloader()
    for _ in tqdm(range(100500)):
        inputs, target, emotions = dataloader.get_batch(8)
    print()






