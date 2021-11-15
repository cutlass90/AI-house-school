import os
class Config:
    def __init__(self):
        self.path_to_data = '/home/ubuntu/mtcnn_cropped/test'
        self.checkpoint_dir = 'checkpoints/face_swap_adain'
        self.device = 'cuda:0'
        self.load_checkpoint_path = 'checkpoints/face_swap_adain/weights/latest.pth'
        self.num_workers = 8

        self.batch_size = 6
        self.frame_size = 160
        self.lr = 0.0002
        self.log_n_steps = 100
        self.log_N_images = 6
        self.save_n_steps = 300

        self.encoder_filters = 8
        self.encoder_downsamples = 4

        self.emb_size = 512

        self.decoder_filters = 32
        self.decoder_upsamples = 5

        # loss coeficients
        self.mse = 1
        self.ident_loss = 0.001


opt = Config()
os.makedirs(os.path.join(opt.checkpoint_dir, 'weights'), exist_ok=True)
