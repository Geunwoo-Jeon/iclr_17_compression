import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.models.priors import JointAutoregressiveHierarchicalPriors as Joint

from torch.nn.parameter import Parameter
from models import *


def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net_17(out_channel_N=out_channel_N)
        self.Decoder = PSDecoder(out_channel_N=out_channel_N)
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        self.out_channel_N = out_channel_N

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16,
                                          input_image.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        # def feature_probs_based_sigma(feature, sigma):
        #     mu = torch.zeros_like(sigma)
        #     sigma = sigma.clamp(1e-10, 1e10)
        #     gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        #     probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        #     total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
        #     return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = iclr18_estimate_bits_z(compressed_feature_renorm)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])

        return clipped_recon_image, mse_loss, bpp_feature


class Quantization(nn.Module):
    @staticmethod
    def forward(input):
        return torch.round(input)

    @staticmethod
    def backward(grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryGMMCompressor(Joint):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(N * 4, 640, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(640, 640, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(640, N * 9, 3, padding=1),
        )

        self.context_prediction = MaskedConv2d(
            N, 2 * N, kernel_size=5, padding=2, stride=1
        )

        self.Quantization = Quantization()

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = self.Quantization(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        self.w_1, self.s_1, self.m_1, self.w_2, self.s_2, self.m_2, self.w_3, self.s_3, self.m_3 \
            = gaussian_params.chunk(9, 1)
        weights = torch.stack([self.w_1, self.w_2, self.w_3], axis=0)
        weights = torch.nn.functional.softmax(weights, dim=0)
        for i in [1, 2, 3]:
            setattr(self, f's_{i}', abs(getattr(self, f's_{i}')))  # abs for scale
            _, temp_likelihoods = self.gaussian_conditional(y, getattr(self, f's_{i}'), means=getattr(self, f'm_{i}'))
            temp_likelihoods = temp_likelihoods * weights[i-1]
            if i == 1:
                y_likelihoods = temp_likelihoods
            else:
                y_likelihoods += temp_likelihoods

        x_hat = self.g_s(y_hat)
        x_hat_clip = x_hat.clamp(0., 1.)
        x_hat_clip_train = x_hat.clamp(-0.5, 1.5)

        mse_loss = torch.mean((x_hat_clip_train - x).pow(2))

        def estimate_bits(likelihoods):
            likelihoods = likelihoods.clamp(1e-5, 1)
            pixel_bits = torch.log(likelihoods) / -math.log(2.0)
            total_bits = pixel_bits.sum()
            return total_bits

        z_bits = estimate_bits(z_likelihoods)
        y_bits = estimate_bits(y_likelihoods)
        n, _, h, w = x.size()
        num_pixels = n * h * w
        bpp_loss = (z_bits + y_bits) / num_pixels

        return x_hat_clip, mse_loss, bpp_loss
