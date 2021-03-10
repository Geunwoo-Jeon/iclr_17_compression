import torch.nn as nn
import torch
from .GDN import GDN
import math
import copy

class BinaryInputEncoder(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192):
        super(BinaryInputEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.Bconv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        self.Bconv1.weight.data.copy_(self.conv1.weight.data)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.Bconv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.Bconv2.weight.data.copy_(self.conv2.weight.data)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, bias=False)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        # torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        # self.gdn3 = GDN(out_channel_N)
        # self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        # torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        # torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def binarization(self):
        for i, weight in enumerate([self.conv1.weight, self.conv2.weight]):
            n = weight[0][0].nelement()
            s = weight.size()
            m = weight.sum(3, keepdim=True).sum(2, keepdim=True).div(n).expand(s)
            if i == 0:
                self.Bconv1.weight.data = weight.sign().mul(m)
            else:
                self.Bconv2.weight.data = weight.sign().mul(m)

    def updateGrad(self):
        biGradlist = [self.Bconv1.weight.grad.data, self.Bconv2.weight.grad.data]
        weightlist = [self.conv1.weight.data, self.conv2.weight.data]
        i = 0
        for biGrad, weight in zip(biGradlist, weightlist):
            signGrad = torch.empty_like(weight).copy_(weight)
            signGrad[weight.lt(-1.0)] = 0
            signGrad[weight.gt(1.0)] = 0
            n = weight[0][0].nelement()
            s = weight.size()
            m = weight.sum(3, keepdim=True).sum(2, keepdim=True).div(n).expand(s)
            grad1 = biGrad.data.mul(signGrad).mul(m)
            grad2 = biGrad.mul(weight.sign()).sum(3, keepdim=True).sum(2, keepdim=True).div(n).expand(s).mul(weight.sign())
            if i == 0:
                self.conv1.weight.grad = copy.deepcopy(biGrad)
                self.conv1.weight.grad.data = grad1.add(grad2).mul(n)
            else:
                self.conv2.weight.grad = copy.deepcopy(biGrad)
                self.conv2.weight.grad.data = grad1.add(grad2).mul(n)
            i = 1

    def forward(self, x):
        x = self.gdn1(self.Bconv1(x))
        x = self.gdn2(self.Bconv2(x))
        x = self.conv3(x)
        return x


def build_model():
        input_image = torch.zeros([4, 3, 256, 256])

        analysis_net = Analysis_net_17()
        feature = analysis_net(input_image)

        print(feature.size())


if __name__ == '__main__':
    build_model()
