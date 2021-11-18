import os
from glob import glob
from os.path import join, basename, dirname
from random import choice

import matplotlib.pyplot as plt
import torch
from imageio import imread
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm

from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda:0'
dataset_path = '~/DATASETS/images/VGG-Face2/data/vggface2_test/test'


def crop_mtcnn(path_to_data, path_to_save):
    os.makedirs(path_to_save, exist_ok=True)
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    for path in tqdm(sorted(glob(join(path_to_data, '*/*.jpg')))):
        dir_name = dirname(path).split('/')[-1]
        os.makedirs(join(path_to_save, dir_name), exist_ok=True)
        with torch.no_grad():
            try:
                mtcnn(pil_loader(path), save_path=join(path_to_save, dir_name, basename(path)))
            except Exception as e:
                print(e, path)

def calc_face_distance(emb1, emb2):
    return (emb1 - emb2).norm()


def img2tensor(img, device):
    return torch.from_numpy(img).div(127.5).add(-1).to(device).permute(2, 0, 1)


def read_image_as_tensor(path, device):
    return img2tensor(imread(path), device)


def check_distributions(path_to_cropped, n_samples):
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device).requires_grad_(False)
    dirs = glob(join(path_to_cropped, '*'))
    same_dist, diff_dist = [], []
    for i in tqdm(range(n_samples)):
        current_dir = choice(dirs)
        path1 = choice(glob(join(current_dir, '*.jpg')))
        path2 = path1
        while path2 == path1:
            path2 = choice(glob(join(current_dir, '*.jpg')))
        new_dir = current_dir
        while new_dir == current_dir:
            new_dir = choice(dirs)
        path3 = choice(glob(join(new_dir, '*.jpg')))
        batch = torch.stack([read_image_as_tensor(p, device) for p in (path1, path2, path3)])
        embs = resnet(batch)
        same_dist.append(calc_face_distance(embs[0], embs[1]).item())
        diff_dist.append(calc_face_distance(embs[0], embs[2]).item())
    plt.hist(same_dist, bins=100, fc=(0, 0, 1, 0.5))
    plt.hist(diff_dist, bins=100, fc=(0, 1, 0, 0.5))
    plt.savefig('./facenet_distribution.jpg')


if __name__ == "__main__":
    crop_mtcnn(path_to_data='/datasets/VGG-Face2_new/VGG-Face2/data/train',
               path_to_save='/datasets/VGG-Face2_new/mtcnn_cropped/train')
    #check_distributions(path_to_cropped='/home/nazar/DATASETS/images/VGG-Face2/mtcnn_cropped/test',
    #                    n_samples=10000)
