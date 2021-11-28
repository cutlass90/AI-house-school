import os
import random
from functools import partial
from glob import glob
from os.path import join, basename
import pickle
import os

import itertools
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
from demo_autoencoder import load_checkpoints

from config import opt
from torchvision.utils import make_grid



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
        inputs_b, targets_b, emotions_b = [], [], []
        for b in range(batch_size):
            noise = torch.randn(1, 512, device=opt.device)
            emo = random.choice(opt.emotion_list)
            self.model.load_checkpoint(join(opt.path_to_stylegan_checkpoints, emo))
            data = self.model.inference(1, noise)[1:]
            inputs_b.append(data[0])
            targets_b.append(data[1])
            emotions_b.append(emo)
        return torch.cat(inputs_b).add(1).div(2), torch.cat(targets_b).add(1).div(2), emotions_b



def main():
    dataloader = Dataloader()

    generator, kp_detector = load_checkpoints(config_path='config/vox-adv-256.yaml', checkpoint_path='/home/nazar/ai house school/emotion_swap/first-order-model/fomm_checkpoints/vox-adv-cpk.pth.tar',
                                              device=opt.device)
    kp_detector_trainable = copy.deepcopy(kp_detector)
    kp_detector_trainable = kp_detector_trainable.requires_grad_(True).train()#todo add adain target emotion

    optimizer = Adam(kp_detector_trainable.parameters(), lr=opt.lr)
    for step in tqdm(itertools.count()):
        inputs, target, emotions = dataloader.get_batch(opt.batch_size)
        with torch.no_grad():
            target_kp = kp_detector(target)
        pred_kp = kp_detector_trainable(inputs)
        loss = sum([F.l1_loss(pred_kp[k], target_kp[k]) for k in target_kp.keys()])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%opt.n_write_log == 0:
            writer.add_scalar('loss', loss.item(), step)

            with torch.no_grad():
                kp_source = kp_detector(inputs)  # x10 affine
                out_GT = generator(inputs, kp_source=kp_source, kp_driving=target_kp)['prediction']
                out_pred = generator(inputs, kp_source=kp_source, kp_driving=pred_kp)['prediction']
                images = torch.cat([inputs, target, out_GT, out_pred])
                grid = make_grid(images, nrow=opt.batch_size, padding=0)
                writer.add_image('inputs__target__out_GT__out_pred', grid, step)
                print()




if __name__ == "__main__":
    main()






