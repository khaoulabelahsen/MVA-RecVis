import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torchvision.models as models
from torch.optim import lr_scheduler

import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch

from data import data_transforms 
from model import model 

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")


path_read = '/content/drive/My Drive/recvis19_a3-master/bird_dataset'
path_write = '/content/drive/My Drive/recvis19_a3-master/experiment'
outfile = '/content/drive/My Drive/recvis19_a3-master/experiment/kaggle.csv'
seed = 1
momentum = 0.5
batch_size = 64
epochs = 10
lr = 0.1
log_interval = 10

use_cuda = torch.cuda.is_available()

state_dict = torch.load('/content/gdrive/My Drive/recvis/experiment/model_12.pth')
model = models.resnet101(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,20)
model.load_state_dict(state_dict)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')


test_dir = path_read + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open('/content/drive/My Drive/recvis19_a3-master/experiment/kaggle101neww.csv', "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + outfile + ', you can upload this file to the kaggle competition website')