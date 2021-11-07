import os
class Config:
    def __init__(self):
        self.train_video_dir = '/home/nazar/datasets/reface_kaggle_v1/train'
        self.test_video_dir = '/home/nazar/datasets/reface_kaggle_v1/test'
        self.train_labels_path = '/home/nazar/datasets/reface_kaggle_v1/train.csv'
        self.checkpoint_dir = 'checkpoints/resnet18'
        self.device = 'cuda:0'
        self.load_checkpoint_path = 'checkpoints/resnet18/epoch33_f1=0.92639.pth'

        self.batch_size = 16
        self.frame_size = 256
        self.val_part = 0.2
        self.n_epoch = 40
        self.lr = 0.0002
        self.max_frames = 300
        self.scoring_everyN_epoch = 3

config = Config()
os.makedirs(os.path.join(config.checkpoint_dir, 'weights/latest'), exist_ok=True)
