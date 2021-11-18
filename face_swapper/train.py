import os
import random
from functools import partial
from glob import glob
from os.path import join, basename
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from imageio import mimread
from sklearn.metrics import f1_score as measure_f1_score
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from networks import Swapper, Vgg19
from tqdm import tqdm
from facenet_pipeline import read_image_as_tensor, calc_face_distance
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision
from losses import PerceptualLoss

from config import opt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

writer = SummaryWriter(log_dir=os.path.join(opt.checkpoint_dir, 'tf_log'))

class VGGDataset(Dataset):
    def __init__(self, path_to_data):
        self.paths = glob(join(path_to_data, '*/*.jpg')) + glob("/datasets/VGG-Face2_new/mtcnn_cropped/train/*/*.jpg")
        print(f'{len(self.paths)} images were found')

    def __getitem__(self, item):
        return read_image_as_tensor(self.paths[item], 'cpu')

    def __len__(self):
        return len(self.paths)


def main():
    vgg19_pretrain = Vgg19().to(opt.device).requires_grad_(False)
    perceptual_loss = PerceptualLoss(criterion=F.l1_loss, feature_extractor=vgg19_pretrain, layer_weights={3:1, 2:1/2, 1:1/4, 0:1/8})
    dataset = VGGDataset(path_to_data=opt.path_to_data)
    face_recognition = InceptionResnetV1(pretrained='vggface2').eval().to(opt.device).requires_grad_(False)
    swapper = Swapper(opt).to(opt.device)
    if opt.load_checkpoint_path:
        swapper.load_state_dict(torch.load(opt.load_checkpoint_path))
        print(f'weight were loaded {opt.load_checkpoint_path}')
    optimizer = Adam(params=swapper.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for epoch in range(100500):
        dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers)
        for step, img in tqdm(enumerate(dataloader)):
            step = 931000 + step + len(dataloader)*epoch
            img = img.to(opt.device)
            with torch.no_grad():
                target_ident_emb = face_recognition(torch.cat([img[1:], img[:1]])) #batch x 512
            swapped = swapper(img, target_ident_emb)

            losses = {
                'perceptual_l1': perceptual_loss(swapped, img),
                'mse': F.mse_loss(swapped, img),
                'ident_loss':  max(0, calc_face_distance(target_ident_emb, face_recognition(swapped)) - 0.8)
            }
            loss = sum(l * getattr(opt, name) for name, l in losses.items())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%opt.log_n_steps == 0:
                for name, l in losses.items():
                    writer.add_scalar(name, l.item(), step)
                    writer.add_scalar("koef/"+name, getattr(opt, name), step)

                N = opt.log_N_images
                im_grid = torchvision.utils.make_grid(torch.cat([img[:N], swapped[:N]], dim=0), normalize=True, range=(-1,1), nrow=N)
                writer.add_image('input_@_swapped', im_grid, step)

            if step%opt.save_n_steps == 0:
                torch.save(swapper.state_dict(), join(opt.checkpoint_dir, 'weights/latest.pth'))










if __name__ == "__main__":
    main()
