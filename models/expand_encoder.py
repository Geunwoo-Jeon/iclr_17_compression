#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import torch.nn as nn
import torch
from .GDN import GDN
import math

class expandEncoder(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192):
        super(expandEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, bias=False)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        return x


def build_model():
        input_image = torch.zeros([4, 3, 256, 256])

        analysis_net = Analysis_net_17()
        feature = analysis_net(input_image)

        print(feature.size())


if __name__ == '__main__':
    build_model()
