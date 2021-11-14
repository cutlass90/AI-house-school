import torch
import torch.nn.functional as F
from torch import nn

from config import opt


class FaceEncoder2(nn.Module):

    def __init__(self, filters, img_size, emb_size, downsamples):
        super(FaceEncoder2, self).__init__()
        self.filters = filters
        self.img_size = img_size
        self.emb_size = emb_size
        self.downsamples = downsamples
        for i in range(self.downsamples):
            inp = 3 if i == 0 else self.filters * 2 ** (i - 1)
            out = self.filters * 2 ** i
            net = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.1)
            )
            setattr(self, 'conv{}'.format(i), net)

        self.dense = nn.Linear(int(self.filters * 2 ** i * (self.img_size / 2 ** (i + 1)) * (self.img_size / 2 ** (i + 1))), self.emb_size)
        print()

    def forward(self, x):
        for i in range(self.downsamples):
            net = getattr(self, 'conv{}'.format(i))
            x = net(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


class FaceDecoder2(nn.Module):

    def __init__(self, emb_size, filters, upsamples, img_size):
        super(FaceDecoder2, self).__init__()
        print('FaceDecoder2')
        self.emb_size = emb_size
        self.filters = filters
        self.upsamples = upsamples
        self.img_size = img_size

        # norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        norm_layer = nn.BatchNorm2d
        out = filters * 2 ** upsamples
        self.dense = nn.Linear(emb_size, out * 4 * 4)

        for k, i in enumerate(list(range(self.upsamples))[::-1]):
            inp = self.filters * 2 ** (i + 1)
            out = self.filters * 2 ** i
            net = nn.Sequential(
                nn.ConvTranspose2d(inp, out, kernel_size=4, stride=2, padding=1),
                norm_layer(out),
                nn.LeakyReLU(0.1)
            )
            print(f'{inp}x{4 * 2 ** k}x{4 * 2 ** k} -> {out}x{4 * 2 ** (k + 1)}x{4 * 2 ** (k + 1)}')
            setattr(self, 'conv{}'.format(i), net)

        self.conv_rgb = nn.Sequential(
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1),
            norm_layer(out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        print(f'Final shape is 3x{4 * 2 ** (k + 1)}x{4 * 2 ** (k + 1)}')

    def forward(self, x):
        x = self.dense(x)
        out = self.filters * 2 ** self.upsamples
        x = x.view(-1, out, 4, 4)
        for i in list(range(self.upsamples))[::-1]:
            net = getattr(self, 'conv{}'.format(i))
            x = net(x)
        x = F.interpolate(x, size=[self.img_size, self.img_size], mode='bilinear')
        x = self.conv_rgb(x)
        return x


class Swapper(nn.Module):
    def __init__(self, opt):
        super(Swapper, self).__init__()
        self.encoder = FaceEncoder2(filters=opt.encoder_filters,
                           img_size=opt.frame_size,
                           emb_size=opt.emb_size,
                           downsamples=opt.encoder_downsamples)

        self.decoder = FaceDecoder2(emb_size=opt.emb_size*2,
                           filters=opt.decoder_filters,
                           upsamples=opt.decoder_upsamples,
                           img_size=opt.frame_size)

    def forward(self, img, ident_emb):
        encoded = self.encoder(img)
        decoded = self.decoder(torch.cat([ident_emb, encoded], dim=1))
        return decoded



if __name__ == "__main__":
    encoder = FaceEncoder2(filters=opt.encoder_filters,
                           img_size=opt.frame_size,
                           emb_size=opt.emb_size,
                           downsamples=opt.encoder_downsamples).to(opt.device)
    img = torch.randn(2, 3, 160, 160).to(opt.device)
    out = encoder(img)
    print()

    decoder = FaceDecoder2(emb_size=opt.emb_size,
                           filters=opt.decoder_filters,
                           upsamples=opt.decoder_upsamples,
                           img_size=opt.frame_size).to(opt.device)
    pred_img = decoder(out)
    print()

    ident_emb = torch.randn(2,512).to(opt.device)
    swapepr = Swapper(opt).to(opt.device)
    swapped = swapepr(img, ident_emb)
    print()
