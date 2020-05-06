# Adapted from : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F




feature_extract= False
# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms, data_transforms_train, data_transforms_val

#### Importing the datasets 

dataset = '/content/drive/My Drive/recvis19_a3-master/bird_dataset'
batch_size = 20

train_datasets = datasets.ImageFolder(dataset + '/train_images',
                         transform=data_transforms_train)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=1)

val_datasets =  datasets.ImageFolder(dataset + '/val_images',transform=data_transforms_val)
val_loader = torch.utils.data.DataLoader(val_datasets,batch_size=batch_size, shuffle=False, num_workers=1)


dataloaders = {'train':train_loader, 'val':val_loader}
dataset_sizes = {'train':len(train_datasets), 'val':len(val_datasets)}

class_names_train = train_datasets.classes

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net, simple_cnn, alexnet, vgg11, resnet34

model_fit=alexnet()
model=model_fit[0]
input_size=model_fit[1]
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')
    
params_to_update = model.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer= optim.SGD(params_to_update, lr=0.001, momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(epoch):

  model.train()
  correct = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    if use_cuda:
      data, target = data.cuda(), target.cuda()
      optimizer.zero_grad()
      output = model(data)
      criterion = torch.nn.CrossEntropyLoss(reduction='mean')
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).cpu().sum()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data.item()), 'Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset))) 

def validation():

    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
nclasses = 20
data = '/content/gdrive/My Drive/recvis/bird_dataset'
batch_size =20
epoch =  20
lr =  0.1
momentum = 0.9
seed = 1
log_interval = 10
experiment = '/content/gdrive/My Drive/recvis/experiment'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)

# Create experiment folder
if not os.path.isdir(experiment):
    os.makedirs(experiment)

# Data initialization and loading
#from data import data_transforms

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

from model import model 

name = 'resnet101'
model = model(name=name)
'''
for name, child in model.named_children():
   if name in ['fc','layer4','layer3']:
       print(name + ' is unfrozen')
       for param in child.parameters():
           param.requires_grad = True
   else:
       print(name + ' is frozen')
       for param in child.parameters():
           param.requires_grad = False
'''

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#best_acc = 0
for epoch in range(1, epoch + 1):
    train(epoch)
    validation()
    #acc = 100. * correct / len(val_loader.dataset)
    #if acc > best_acc : 
      #best_acc = acc
      #best_model = copy.deepcopy(model.state_dict())     
    model_file = experiment + '/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')

#model.load_state_dict(best_model)