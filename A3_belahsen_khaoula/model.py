# Adapted from : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#import torchvision.models as models 
nclasses = 20


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def simple_cnn():
    return SimpleCNN(), (64, 64)


def alexnet():
    model = torch_alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, nclasses),
    )
    return model, (224, 224)


def resnet34(nclass=None):
    model = torch_resnet34(pretrained=True)
    model_conv = nn.Sequential(*list(model.children())[:-3])
    for param in model_conv.parameters():
        param.requires_grad = False
    fc_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_features, nclass if nclass is not None else nclasses),
        nn.Softmax(dim=-1)
    )
    return model, (224, 224)


def vgg11(nclass=None):
    model = torch_vgg11(pretrained=True)
    model_conv = nn.Sequential(*list(model.children())[:-1])
    for param in model_conv.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, nclasses),
    )
    return model, (224, 224)


def model(name):
  if name =='vgg19':
    model_ft = models.vgg19(pretrained='imagenet')
    model_conv = nn.Sequential(*list(model.children())[:-1])
    for param in model_conv.parameters():
        param.requires_grad = False
    model_ft.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, nclasses),
    )

  else:
    if name=='resnet152':
      model_ft = models.resnet152(pretrained=True)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs, 20)
    if name=='resnet101':
      model_ft = models.resnet101(pretrained=True)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs, 20)
    elif name=='resnet18':
      model_ft = models.resnet18(pretrained=True)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs, 20)
    else:
      model_ft = models.resnet50(pretrained=True)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs, 20)
      
  return model_ft.to(device)
