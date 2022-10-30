import torch
import pickle
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import time
import torchvision
import torchvision.transforms as transforms
from torchvision.models import *

class CIFAR100Test(Dataset):
  def __init__(self, path, transform=None):
    with open(os.path.join(path, 'test'), 'rb') as cifar100:
      self.data = pickle.load(cifar100, encoding='bytes')
    self.transform = transform

  def __len__(self):
    return len(self.data['data'.encode()])

  def __getitem__(self, index):
    label = self.data['fine_labels'.encode()][index]
    r = self.data['data'.encode()][index, :1024].reshape(32, 32)
    g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
    b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
    image = np.dstack((r, g, b))

    if self.transform:
      image = self.transform(image)

    return label, image

""" failed models """
# net = googlenet(pretrained=True)
# net = resnet18(pretrained=True)
# net = inception_v3(pretrained=True)

""" successful models """
# net = vgg11(pretrained=True)
# net = mobilenet_v2(pretrained=True)
# net = densenet121(pretrained=True)
# net = squeezenet1_0(pretrained=True)
# net = mnasnet0_5(pretrained=True)
# net = shufflenet_v2_x0_5(pretrained=True)
net = alexnet(pretrained=True)  # add transforms.Resize() to transforms.Compose
net.eval()
cifar_path = '/media/bst/hdd1/mirae/pytorch-cifar100/data'
# imagenet_path = '/media/bst/hdd1/mirae/pytorch-imagenet/data'

if net._get_name() == 'AlexNet':
  compose_list = [
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.486548, 0.44091], std=[0.2673, 0.25643, 0.27615])
  ]
else:
  compose_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.486548, 0.44091], std=[0.2673, 0.25643, 0.27615])
  ]

transform_test = transforms.Compose(compose_list)
# test_loader = DataLoader(CIFAR100Test(cifar_path, transform_test))
cifar_test = torchvision.datasets.CIFAR100(root=cifar_path, train=False, download=False, transform=transform_test)
# cifar_test = torchvision.datasets.ImageNet(root=imagenet_path, train=False, download=False, transform=transform_test)
test_loader = DataLoader(cifar_test, shuffle=True)

correct_1 = 0.0
correct_5 = 0.0
latency = []

with torch.no_grad():
  # net.to('cuda:0')
  net.hetero()
  print(net)
  print('\n--------------------\n')
  for n_iter, (image, label) in enumerate(test_loader):
    print('iteration: {}\ttotal {} iterations'.format(n_iter + 1, len(test_loader)))

    # image = image.cuda()
    # label = label.cuda()

    start = time.time()
    output = net(image)
    end = time.time()
    print('elapsed time: {}'.format(end - start))
    latency.append(end - start)
    _, pred = output.topk(5, 1, largest=True, sorted=True)

    label = label.view(label.size(0), -1).expand_as(pred)
    correct = pred.eq(label).float()

    correct_5 += correct[:, :5].sum()
    correct_1 += correct[:, :1].sum()

    if n_iter == 100:
      break

print('Top 1 err: ', 1 - correct_1 / len(test_loader.dataset))
print('Top 5 err: ', 1 - correct_5 / len(test_loader.dataset))
print('Average latency: {}'.format(np.average(latency)))
print('Median latency: {}'.format(np.median(latency)))
print('P99 latency: {}'.format(np.percentile(latency, 99)))
print('Parameter numbers: {}'.format(sum(p.numel() for p in net.parameters())))


'''
class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    # 1 input image channel, 6 output channels, 3x3 square conv kernel
    self.conv1 = nn.Conv2d(1, 6, 3)
    self.conv2 = nn.Conv2d(6, 16, 3)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
    x = x.view(-1, int(x.nelement() / x.shape[0]))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

model = LeNet()
model.eval()
model.hetero()
'''
