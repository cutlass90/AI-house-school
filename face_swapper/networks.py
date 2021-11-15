import torch
import torch.nn.functional as F
from torch import nn

from config import opt


class AdaIN(nn.Module):
    def __init__(self, channels, latent_size):
        super().__init__()
        self.channels = channels
        self.linear = nn.Sequential(nn.Linear(latent_size, (channels+latent_size)//2),
                                    nn.ELU(),
                                    nn.Linear((channels+latent_size)//2, channels*2))

    def forward(self, x, dlatent):
        x = nn.InstanceNorm2d(self.channels)(x)
        style = self.linear(dlatent)
        style = style.view([-1, 2, x.size()[1]] + [1] * (len(x.size()) - 2))
        return x * (style[:, 0] + 1) + style[:, 1]


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
            setattr(self, 'adain{}'.format(i), AdaIN(channels=out, latent_size=512))
            print(f'{inp}x{4 * 2 ** k}x{4 * 2 ** k} -> {out}x{4 * 2 ** (k + 1)}x{4 * 2 ** (k + 1)}')
            setattr(self, 'conv{}'.format(i), net)

        self.conv_rgb = nn.Sequential(
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1),
            norm_layer(out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        print(f'Final shape is 3x{4 * 2 ** (k + 1)}x{4 * 2 ** (k + 1)}')

    def forward(self, x, iden_emb):
        x = self.dense(x)
        out = self.filters * 2 ** self.upsamples
        x = x.view(-1, out, 4, 4)
        for i in list(range(self.upsamples))[::-1]:
            net = getattr(self, 'conv{}'.format(i))
            x = net(x)
            x = getattr(self, 'adain{}'.format(i))(x, iden_emb)
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
        decoded = self.decoder(torch.cat([ident_emb, encoded], dim=1), ident_emb)
        return decoded



if __name__ == "__main__":
    encoder = FaceEncoder2(filters=opt.encoder_filters,
                           img_size=opt.frame_size,
                           emb_size=opt.emb_size,
                           downsamples=opt.encoder_downsamples).to(opt.device)
    img = torch.randn(2, 3, 160, 160).to(opt.device)
    out = encoder(img)
    print()

    ident_emb = torch.randn(2,512).to(opt.device)
    decoder = FaceDecoder2(emb_size=opt.emb_size,
                           filters=opt.decoder_filters,
                           upsamples=opt.decoder_upsamples,
                           img_size=opt.frame_size).to(opt.device)
    pred_img = decoder(out, ident_emb)
    print()

    swapepr = Swapper(opt).to(opt.device)
    swapped = swapepr(img, ident_emb)
    print()
