import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable



train_data = torchvision.datasets.FashionMNIST('./data',
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose(
                                                      [
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))
                                                      ])
                                                  )

valid_data = torchvision.datasets.FashionMNIST('./data',
                                               train=True,
                                               download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,))
                                               ])
                                               )

train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)

train_data.train_data = train_data.train_data[train_idx, :]
train_data.train_labels = train_data.train_labels[torch.from_numpy(train_idx).type(torch.LongTensor)]

mask = np.ones(60000)
mask[train_idx] = 0

valid_data.train_data = valid_data.train_data[torch.from_numpy(np.argwhere(mask)), :].squeeze()
valid_data.train_labels = valid_data.train_labels[torch.from_numpy(mask).type(torch.ByteTensor)]


batch_size = 100
test_batch_size = 100

train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_data,
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)

plt.imshow(train_loader.dataset.train_data[1].numpy())
plt.show()
