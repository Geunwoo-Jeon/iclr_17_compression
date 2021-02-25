import os
import argparse
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json

parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

parser.add_argument('-n', '--name', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--config', dest='config', required=False,
        help = 'hyperparameter in json format')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')

args = parser.parse_args()
model = ImageCompressor()
load_model(model, args.pretrain)
print(model.Encoder.Bconv2.weight.data[0][0:3])
