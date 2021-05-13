from .analysis_17 import Analysis_net_17
import torch.nn as nn
from .GDN import GDN
import torch
import math


class PSDecoder(nn.Module):
    '''
    Decode synthesis
    '''

    def __init__(self, out_channel_N=192):
        super(PSDecoder, self).__init__()
        self.conv1 = nn.Conv2d(192, 192, 5, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * 1 )))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.igdn1 = GDN(192, inverse=True)
        self.conv2 = nn.Conv2d(48, 48, 5, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.igdn2 = GDN(48, inverse=True)
        self.conv3 = nn.Conv2d(12, 48, 9, padding=4)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.PS2 = nn.PixelShuffle(2)
        self.PS4 = nn.PixelShuffle(4)

    def forward(self, x):
        x = self.PS2(self.igdn1(self.conv1(x)))
        x = self.PS2(self.igdn2(self.conv2(x)))
        x = self.PS4(self.conv3(x))
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
