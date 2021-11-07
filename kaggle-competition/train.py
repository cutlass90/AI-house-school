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
from torchvision.models.resnet import resnet18
from tqdm import tqdm

from config import config as opt

measure_f1_score = partial(measure_f1_score, average='micro')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

writer = SummaryWriter(log_dir=os.path.join(opt.checkpoint_dir, 'tf_log'))


class VideoFakeDataset(Dataset):

    def __init__(self, video_paths, label_csv_path=None, frame_size=256, mode='train'):
        self.label_csv_path = label_csv_path
        self.video_paths = video_paths
        if label_csv_path is not None:
            self.labels = pd.read_csv(label_csv_path)
        self.frame_size = frame_size
        self.mode = mode

    def __getitem__(self, id):
        sample = {'filename': basename(self.video_paths[id])}

        frames = mimread(self.video_paths[id], memtest=False)
        if self.mode == 'train':
            frame = torch.from_numpy(random.choice(frames)).div(127.5).sub(1).permute(2, 0, 1)
            if frame.size(1) != opt.frame_size:
                frame = F.interpolate(frame.unsqueeze(0), size=[self.frame_size, self.frame_size], mode='bilinear', align_corners=False)[0]  # todo check image size
        elif self.mode == 'eval':
            frame = torch.from_numpy(np.array(frames)).div(127.5).sub(1).permute(0, 3, 1, 2)
            if frame.size(2) != opt.frame_size:
                frame = F.interpolate(frame, size=[self.frame_size, self.frame_size], mode='bilinear', align_corners=False)
        sample['frames'] = frame

        if self.label_csv_path:
            sample['labels'] = torch.tensor(float(self.labels.label.loc[self.labels.filename == basename(self.video_paths[id])]))
        return sample

    def __len__(self):
        return len(self.video_paths)


def train_epoch(epoch, model, dataloader, optimizer):
    print(f'start {epoch} epoch')
    for step, batch in tqdm(enumerate(dataloader)):
        frames = batch['frames'].to(opt.device)
        labels = batch['labels'].to(opt.device).unsqueeze(1)
        predicted = model(frames)
        loss = F.binary_cross_entropy_with_logits(predicted, labels)
        writer.add_scalar('loss', loss.item(), global_step=epoch * len(dataloader) + step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def score_model(model, dataloader):
    """

    :param model:
    :param dataloader:
    :return:
        res: dict, {filename: predicted value}
    """
    print('Model scoring was started')
    model.eval()
    dataloader.dataset.mode = 'eval'

    res = {}
    with torch.no_grad():
        for batch in tqdm(dataloader.dataset):
            frames = batch['frames'][:opt.max_frames].to(opt.device)
            predicted = (torch.sigmoid(model(frames)).mean() > 0.5).int()
            res[batch['filename']] = predicted.cpu().numpy().item()
    if dataloader.dataset.label_csv_path is not None:
        predicted = [res[filename] for filename in sorted(res.keys())]
        target = [int(dataloader.dataset.labels.label.loc[dataloader.dataset.labels.filename == filename]) for filename in sorted(res.keys())]
        f1_score = measure_f1_score(target, predicted)
    else:
        f1_score = None

    model.eval()
    dataloader.dataset.mode = 'train'
    return f1_score, res


def train_model():
    paths = glob(join(opt.train_video_dir, '*.mp4'))
    print(f'{len(paths)} train files were found')
    random.shuffle(paths)
    n_train_samples = int(len(paths) * (1 - opt.val_part))

    if os.path.isfile('train_paths.pkl'):
        with open('train_paths.pkl', 'rb') as f:
            train_paths = pickle.load(f)
    else:
        train_paths = paths[:n_train_samples]
        with open('train_paths.pkl', 'wb') as f:
            pickle.dump(train_paths, f)

    if os.path.isfile('test_paths.pkl'):
        with open('test_paths.pkl', 'rb') as f:
            test_paths = pickle.load(f)
    else:
        test_paths = paths[n_train_samples:]
        with open('test_paths.pkl', 'wb') as f:
            pickle.dump(test_paths, f)



    train_dataloader = DataLoader(dataset=VideoFakeDataset(video_paths=train_paths, label_csv_path=opt.train_labels_path, frame_size=opt.frame_size),
                                  batch_size=opt.batch_size, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(dataset=VideoFakeDataset(video_paths=test_paths, label_csv_path=opt.train_labels_path, frame_size=opt.frame_size),
                                batch_size=opt.batch_size, shuffle=True, num_workers=12)

    if opt.load_checkpoint_path:
        model = torch.load(opt.load_checkpoint_path).to(opt.device)
        print(f'model was loaded from {opt.load_checkpoint_path}')
    else:
        model = resnet18(pretrained=False, num_classes=1).to(opt.device)

    optimizer = Adam(model.parameters(), lr=opt.lr)
    for epoch in range(15, opt.n_epoch):
        model = train_epoch(epoch, model, train_dataloader, optimizer)
        torch.save(model, os.path.join(opt.checkpoint_dir, 'weights/latest/latest.pth'))
        if epoch % opt.scoring_everyN_epoch == 0:
            f1_score, _ = score_model(model, val_dataloader)
            writer.add_scalar('val_f1', f1_score, (epoch + 1) * len(train_dataloader))
            torch.save(model, join(opt.checkpoint_dir, f'epoch{epoch}_f1={round(f1_score, 5)}.pth'))

            # f1_score, _ = score_model(model, train_dataloader)
            # writer.add_scalar('train_f1', f1_score, (epoch + 1) * len(train_dataloader))
    return model


def make_submission_file(model):
    video_paths = glob(join(opt.test_video_dir, '*.mp4'))
    test_dataloader = DataLoader(VideoFakeDataset(video_paths=video_paths, label_csv_path=None, frame_size=opt.frame_size),
                              batch_size=opt.batch_size, shuffle=True, num_workers=8)
    _, predictions = score_model(model, test_dataloader)
    with  open(join(opt.checkpoint_dir, 'submission.scv'), 'w') as f:
        f.writelines(['filename,label\n'])
        for filename, label in predictions.items():
            f.writelines([f'{filename},{int(label)}\n'])



def main():
    model = train_model()
    # model = torch.load(opt.load_checkpoint_path).to(opt.device)
    # print(f'model was loaded from {opt.load_checkpoint_path}')
    # make_submission_file(model)
    # print('DONE')


if __name__ == '__main__':
    main()
