import torch.nn as nn
import torch
from .GDN import GDN
import math
import copy

class BinaryEncoder(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192):
        super(BinaryEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
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
        weight = self.conv2.weight.data
        n = weight[0][0].nelement()
        s = weight.size()
        m = weight.sum(3, keepdim=True).sum(2, keepdim=True).div(n).expand(s)
        self.Bconv2.weight.data = weight.sign().mul(m)

    def updateGrad(self):
        biGrad = self.Bconv2.weight.grad.data
        weight = self.conv2.weight.data
        signGrad = torch.empty_like(weight).copy_(weight)
        signGrad[weight.lt(-1.0)] = 0
        signGrad[weight.gt(1.0)] = 0
        n = weight[0][0].nelement()
        s = weight.size()
        m = weight.sum(3, keepdim=True).sum(2, keepdim=True).div(n).expand(s)
        grad1 = biGrad.data.mul(signGrad).mul(m)
        grad2 = biGrad.mul(weight.sign()).sum(3, keepdim=True).sum(2, keepdim=True).div(n).expand(s).mul(weight.sign())
        self.conv2.weight.grad = copy.deepcopy(biGrad)
        self.conv2.weight.grad.data = grad1.add(grad2).mul(n)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
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
