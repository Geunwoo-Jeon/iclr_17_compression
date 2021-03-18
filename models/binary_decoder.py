from .analysis_17 import Analysis_net_17
import torch.nn as nn
from .GDN import GDN
import torch
import math
import copy


class BinaryDecoder(nn.Module):
    '''
    Decode synthesis
    '''

    def __init__(self, out_channel_N=192):
        super(BinaryDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 )))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.Bdeconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.Bdeconv2.weight.data.copy_(self.deconv2.weight.data)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, 3, 9, stride=4, padding=4, output_padding=3)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def binarization(self):
        weight = self.deconv2.weight.data
        n = weight[0][0].nelement()
        s = weight.size()
        m = weight.sum(3, keepdim=True).sum(2, keepdim=True).div(n).expand(s)
        self.Bdeconv2.weight.data = weight.sign().mul(m)

    def updateGrad(self):
        biGrad = self.Bdeconv2.weight.grad.data
        weight = self.deconv2.weight.data
        signGrad = torch.empty_like(weight).copy_(weight)
        signGrad[weight.lt(-1.0)] = 0
        signGrad[weight.gt(1.0)] = 0
        n = weight[0][0].nelement()
        s = weight.size()
        m = weight.sum(3, keepdim=True).sum(2, keepdim=True).div(n).expand(s)
        grad1 = biGrad.data.mul(signGrad).mul(m)
        grad2 = biGrad.mul(weight.sign()).sum(3, keepdim=True).sum(2, keepdim=True).div(n).expand(s).mul(weight.sign())
        self.deconv2.weight.grad = copy.deepcopy(biGrad)
        self.deconv2.weight.grad.data = grad1.add(grad2).mul(n)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.Bdeconv2(x))
        x = self.deconv3(x)
        return x

# synthesis_one_pass = tf.make_template('synthesis_one_pass', synthesis_net)

def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net_17()
    synthesis_net = Synthesis_net_17()
    feature = analysis_net(input_image)
    recon_image = synthesis_net(feature)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("recon_image : ", recon_image.size())

# def main(_):
#   build_model()


if __name__ == '__main__':
    build_model()
