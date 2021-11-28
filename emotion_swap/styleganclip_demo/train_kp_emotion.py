import itertools
import os
import random
from functools import partial
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score as measure_f1_score
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torchvision.utils import make_grid
from tqdm import tqdm

from config import opt
from demo_autoencoder import load_checkpoints
from modules.keypoint_detector import KPDetector
from stylegan_infer import Model


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
            target_emotion = torch.zeros([1, 7]).to(opt.device)
            target_emotion[:, opt.emotion_list.index(emo)] = 1
            emotions_b.append(target_emotion)
        return torch.cat(inputs_b).add(1).div(2), torch.cat(targets_b).add(1).div(2), torch.cat(emotions_b)


def main():
    dataloader = Dataloader()

    generator, kp_detector = load_checkpoints(config_path='config/vox-adv-256.yaml', checkpoint_path='/home/nazar/ai house school/emotion_swap/first-order-model/fomm_checkpoints/vox-adv-cpk.pth.tar',
                                              device=opt.device)
    kp_detector_trainable = KPDetector(block_expansion=32, num_kp=10, num_channels=3, max_features=1024, num_blocks=5, temperature=0.1, estimate_jacobian=True, scale_factor=0.25,
                                       single_jacobian_map=False, pad=0, adain_size=7).train().requires_grad_(True).to(opt.device)
    kp_detector_trainable.load_state_dict(kp_detector.state_dict(), strict=False)
    kp_detector_trainable = kp_detector_trainable.requires_grad_(True).train()

    optimizer = Adam(kp_detector_trainable.parameters(), lr=opt.lr)
    for step in tqdm(itertools.count()):
        inputs, target, emotions = dataloader.get_batch(opt.batch_size)
        with torch.no_grad():
            target_kp = kp_detector(target)
        pred_kp = kp_detector_trainable(inputs, emotions)
        loss = sum([F.l1_loss(pred_kp[k], target_kp[k]) for k in target_kp.keys()])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % opt.n_write_log == 0:
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
