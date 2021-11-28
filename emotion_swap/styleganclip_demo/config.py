import os


class Config:
    def __init__(self):
        self.path_to_stylegan_checkpoints = '/home/nazar/ai house school/emotion_swap/styleganclip_demo/stylegan_checkpoints'
        self.emotion_list = [
            'angry',
            'disgust',
            'fear',
            'happy',
            'neutral',
            'sad',
            'surprise'
        ]
        self.checkpoint_dir = 'checkpoints/version1'
        self.device = 'cuda:0'
        self.load_checkpoint_path = ''

        self.batch_size = 4


opt = Config()
os.makedirs(os.path.join(opt.checkpoint_dir, 'weights'), exist_ok=True)
