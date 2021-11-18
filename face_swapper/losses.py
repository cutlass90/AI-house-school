from functools import reduce
import random

import numpy as np
import torch
import torch.autograd as autograd
from cvxopt import matrix, spmatrix, solvers, sparse
from torch import nn
# from reflect_cv_tools.nn.utils import random_crop_tensors
# from ..augmentation.differentiable import make_augment
from torch.nn.functional import unfold


class WGAN_QC:
    def __init__(self, netD, optimizerD, batchSize, K, Diters=1, gamma=0.05, DOptIters=1, alpha=0):
        """
        Args:
            netD: torch.nn.Module, network must predict logits for each sample in batch in rage [-inf, +inf]
            batchSize: int
            Diters: int, number of D iters to one iteration of generator
            K: the coef in transport cost, <=0 meaning K = 1/dim, where dim is channel*width*high
            gamma: float, gamma for optimal transport regularization
            DOptIters: int, number of iters of regression of D
            alpha: float should be equals 0??


        """
        self.netD = netD
        self.optimizerD = optimizerD
        self.batchSize = batchSize
        self.Diters = Diters
        self.K = K
        self.gamma = gamma
        self.DOptIters = DOptIters
        self.alpha = alpha

        self.criterion = nn.MSELoss()
        self.device = next(netD.parameters()).device
        self.Kr = np.sqrt(K)
        self.LAMBDA = 2 * self.Kr * gamma * 2

        ###############################################################################
        ###################### Prepare linear programming solver ######################
        solvers.options['show_progress'] = False
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

        A = spmatrix(1.0, range(batchSize), [0] * batchSize, (batchSize, batchSize))
        for i in range(1, batchSize):
            Ai = spmatrix(1.0, range(batchSize), [i] * batchSize, (batchSize, batchSize))
            A = sparse([A, Ai])

        D = spmatrix(-1.0, range(batchSize), range(batchSize), (batchSize, batchSize))
        DM = D
        for i in range(1, batchSize):
            DM = sparse([DM, D])

        self.A = sparse([[A], [DM]])

        cr = matrix([-1.0 / batchSize] * batchSize)
        cf = matrix([1.0 / batchSize] * batchSize)
        self.c = matrix([cr, cf])

        self.pStart = {}
        self.pStart['x'] = matrix([matrix([1.0] * batchSize), matrix([-1.0] * batchSize)])
        self.pStart['s'] = matrix([1.0] * (2 * batchSize))
        ###############################################################################

    def train_D(self, real, fake, loss_weight=None):
        fake = fake.detach()
        for p in self.netD.parameters():  # reset requires_grad
            p.requires_grad = True

        self.netD.train()

        ###########################################################################
        #                    Deep Regression for discriminator
        ###########################################################################
        ##################### perform deep regression for D #######################
        j = 0
        while j < self.Diters:
            j += 1
            dist = self.comput_dist(real, fake)
            sol = self.Wasserstein_LP(dist)
            if self.LAMBDA > 0:
                mapping = self.approx_OT(sol)
                real_ordered = real[mapping]  # match real and fake
                RF_dif = real_ordered - fake

            # construct target
            target = torch.from_numpy(np.array(sol['x'])).float()
            target = target.squeeze().to(self.device)

            for k in range(self.DOptIters):
                self.netD.zero_grad()
                fake.requires_grad_()
                if fake.grad is not None:
                    fake.grad.data.zero_()
                output_real = self.netD(real)
                output_fake = self.netD(fake)
                output_real, output_fake = output_real.squeeze(), output_fake.squeeze()
                output_R_mean = output_real.mean(0).view(1)
                output_F_mean = output_fake.mean(0).view(1)

                L2LossD_real = self.criterion(output_R_mean[0], target[:self.batchSize].mean())
                L2LossD_fake_1 = self.criterion(output_F_mean[0], target[self.batchSize:].mean())
                L2LossD_fake_2 = self.criterion(output_fake, target[self.batchSize:])
                L2LossD_fake = self.alpha * L2LossD_fake_1 + (1 - self.alpha) * L2LossD_fake_2
                L2LossD = 0.5 * L2LossD_real + 0.5 * L2LossD_fake

                if self.LAMBDA > 0:
                    RegLossD = self.OT_regularization(output_fake, fake, RF_dif)
                    TotalLoss = L2LossD + self.LAMBDA * RegLossD
                else:
                    TotalLoss = L2LossD

                loss = TotalLoss if loss_weight is None else TotalLoss * loss_weight
                loss.backward()
                self.optimizerD.step()

            WD = output_R_mean - output_F_mean  # Wasserstein Distance
        return WD, TotalLoss

    def train_G(self, fake):
        for p in self.netD.parameters():
            p.requires_grad = False  # frozen D
        output_fake = self.netD(fake)
        output_F_mean_after = -1 * output_fake.mean()
        return output_F_mean_after

    def OT_regularization(self, output_fake, fake, RF_dif):
        output_fake_grad = torch.ones(output_fake.size()).to(self.device)
        gradients = autograd.grad(outputs=output_fake, inputs=fake,
                                  grad_outputs=output_fake_grad,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        n = gradients.size(0)
        RegLoss = 0.5 * (
            (gradients.view(n, -1).norm(dim=1) / (2 * self.Kr) - self.Kr / 2 * RF_dif.view(n, -1).norm(dim=1)).pow(
                2)).mean()
        fake.requires_grad = False

        return RegLoss

    def comput_dist(self, real, fake):
        data_dim = reduce(lambda x, y: x * y, real.size()[1:])
        num_r = real.size(0)
        num_f = fake.size(0)
        real_flat = real.view(num_r, -1)
        fake_flat = fake.view(num_f, -1)

        real3D = real_flat.unsqueeze(1).expand(num_r, num_f, data_dim)
        fake3D = fake_flat.unsqueeze(0).expand(num_r, num_f, data_dim)
        # compute squared L2 distance
        dif = real3D - fake3D
        dist = 0.5 * dif.pow(2).sum(2).squeeze()
        K = 1 / data_dim if self.K is None else self.K
        return dist * K

    def Wasserstein_LP(self, dist):
        b = matrix(dist.cpu().double().numpy().flatten())
        sol = solvers.lp(self.c, self.A, b, primalstart=self.pStart, solver='glpk')
        offset = 0.5 * (sum(sol['x'])) / self.batchSize
        sol['x'] = sol['x'] - offset
        self.pStart['x'] = sol['x']
        self.pStart['s'] = sol['s']
        return sol

    def approx_OT(self, sol):
        ################ Compute the OT mapping for each fake data ################
        ResMat = np.array(sol['z']).reshape((self.batchSize, self.batchSize))
        mapping = torch.from_numpy(np.argmax(ResMat, axis=0)).long().to(self.device)
        return mapping


class PerceptualLoss(nn.Module):
    def __init__(self, criterion, feature_extractor, layer_weights, crop_size=None):
        """

        Args:
             criterion: function, loss function that give scalar loss like func(input, target)
             feature_extractor: torch.nn.Module, that gives list of feature tensors
             layer_weights: dict, correspondence of layer and weights for it, example {0: 0.1, 1: 0.4, 2: 1.5}
             crop_size: int, each tensor crop to this size, if None full tensor size is used
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.criterion = criterion
        self.layer_weights = layer_weights
        self.crop_size = crop_size

    def forward(self, input, target):
        input = self.feature_extractor(input)
        with torch.no_grad():
            target = self.feature_extractor(target)
        loss = 0
        for layer, weight in self.layer_weights.items():
            if (self.crop_size is not None) and (input[layer].size()[2] > self.crop_size):
                input_t, target_t = random_crop_tensors([input[layer], target[layer]], self.crop_size)
            else:
                input_t, target_t = input[layer], target[layer]
            loss += weight * self.criterion(input_t, target_t)
        return loss


class ContextualLoss(nn.Module):

    def __init__(self, h=0.1):
        super(ContextualLoss, self).__init__()
        self.h = h

    def forward(self, inputs, targets):
        inputs, targets = self.center_by_T(inputs, targets)
        inputs, targets = self.l2_normalize_channelwise(inputs), self.l2_normalize_channelwise(targets)
        batch_size, dep = inputs.size()[0:2]
        targets = targets.view(batch_size, dep, -1)
        inputs = inputs.view(batch_size, dep, -1)
        inputs = torch.transpose(inputs, 2, 1)
        cosine_dist = torch.bmm(inputs, targets)
        cosine_dist = (1 - cosine_dist) / 2
        relative_dist = self.calc_relative_distances(cosine_dist)
        cs_flow = self.calculate_CS(relative_dist)
        k_max_NC = torch.max(cs_flow, dim=1)[0]
        CS = k_max_NC.mean(1)
        CX_as_loss = 1 - CS
        CX_loss = -torch.log(1 - CX_as_loss)
        CX_loss = torch.mean(CX_loss)
        return CX_loss

    def calc_relative_distances(self, cosine_dist):
        # Note cosine_dist has shape inputs x targets
        epsilon = 1e-5
        div = torch.min(cosine_dist, dim=2, keepdim=True)[0]
        relative_dist = cosine_dist / (div + epsilon)
        return relative_dist

    def calculate_CS(self, relative_dist):
        cs_weights_before_normalization = torch.exp((1 - relative_dist) / self.h)
        s = torch.sum(cs_weights_before_normalization, dim=2, keepdim=True)
        return cs_weights_before_normalization / (s + 1e-8)

    def center_by_T(self, inputs, targets):
        meanT = targets.mean(2).mean(2).mean(0)
        ch = meanT.size()[0]
        meanT = meanT.view(ch, 1, 1)
        return inputs - meanT, targets - meanT

    def l2_normalize_channelwise(self, inputs):
        return inputs / torch.norm(inputs, p=2, dim=1).unsqueeze(1)


def cosine_loss(inputs, targets):
    """ Calculate cosine loss in range between 0 and 2

    Args:
        inputs: torch.tensor of size batch x N
        targets: torch.tensor of size batch x N

    Return:
        torch.tensor of size 1
    """
    cos = 1 - nn.functional.cosine_similarity(inputs, targets)
    return cos.mean()


def augmented_cosine(inputs, targets, identity_estimator, n_augment=4):
    """ Make differentiable augmentation and calculate cosine loss

    Args:
        inputs: torch.tensor of size batch x 3 x H x W
        targets: torch.tensor of size batch x  3 x H x W
        identity_estimator: nn.Module that map image to embedding

    Return:
        torch.tensor of size 1
    """
    reals = []
    swappeds = []
    for i in range(n_augment):
        reals.append(make_augment(targets.flip([3]) if random.random() > 0.5 else targets))
        swappeds.append(make_augment(inputs.flip([3]) if random.random() > 0.5 else inputs))
    targets = torch.cat(reals)
    inputs = torch.cat(swappeds)
    swapped_ident_emb = identity_estimator(inputs)
    with torch.no_grad():
        target_ident_emb = identity_estimator(targets)
    loss = cosine_loss(swapped_ident_emb, target_ident_emb)
    return loss

def kolmagorov_smirnov_loss(input, target, mask=None, patch_size=13, effective_samples=1024):
    """
    Compute Kolmagorov Smirnov loss made by Alexey Chaplygin

    Args:
        input: torch tensor of shape batch 3 height width, your generated image
        target: torch tensor of shape batch 3 height width, target image
        mask: torch tensor of shape batch 1 height width, attention mask of the loss
        patch_size: int, size of attention window, image are splitted to this windows
        effective_samples: int, number of pixels that are randomly sampled during calculating for memory efficiency, None for whole image

    Return:
        torch tensor
    """
    if mask is None:
        mask = torch.ones([input.size(0), 1, input.size(2), input.size(3)]).to(input.device)
    imJ = torch.stack([target, input], dim=1)
    gr = torch.clamp(torch.rand(input.size(0), 1, 3, 1, 1).to(input.device), min=0.01, max=1)
    gr = gr / torch.sum(gr, 2, keepdim=True)
    imJ = torch.cat([torch.sum(imJ * gr, 2), mask], 1).view(-1, 1, input.size(2), input.size(3))  # b*3 1 192 192
    imJ = unfold(imJ, patch_size) #b*3 169 180*180
    imJ = imJ.permute(0,2,1) #b*3 180*180 169
    imJ = imJ.view(input.size(0), 3, imJ.size(1), patch_size**2) #b 3 180*180 * 169
    imJ = imJ.permute(0, 2, 1, 3).reshape(-1, 3, patch_size**2) #100500 3 169
    if effective_samples is not None:
        imJ = imJ[torch.randperm(imJ.size(0))[:effective_samples]] #effective_samples 3 169
    imJ_ma = torch.mean(imJ[:, -1:], dim=2, keepdim=True)
    imJ, _ = torch.sort(imJ[:, :-1], dim=2)
    loss_l1_px = torch.mean(torch.abs(imJ[:, :1] - imJ[:, 1:]) * imJ_ma)
    return loss_l1_px
