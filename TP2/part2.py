import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
import random
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

kernel_size = 5
kernel_size_pooling = 2


class FcNetwork(nn.Module):
    def __init__(self, act_func, nb_layers):
        super().__init__()
        self.nb_layers = nb_layers
        self.act_func = act_func

        self.fcStart = nn.Linear(28 * 28, 512)
        self.fcEnd = nn.Linear(512, 10)

        for i in range(nb_layers - 2):


    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


class Cnn(nn.Module):
    def __init__(self, act_func, nb_layers):
        super().__init__()
        self.act_func = act_func
        self.nb_layers = nb_layers

        self.kernel1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size_pooling, stride=2)
        )

        self.kernel2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size_pooling, stride=2)
        )

        self.layers = []
        self.layers.append(nn.Linear(7 * 7 * 64, 1000))

        lastOutput = 1000

        for i in range(nb_layers-2):
            newOutput = int(random)
            self.layers.append(nn.Linear(lastOutput, ))

        self.layers.append(nn.Linear(1000, 10))
        self.drop_out = nn.Dropout()

    def forward(self, image):
        output = self.kernel1(image)
        output = self.kernel2(output)
        output = output.reshape(output.size(0), -1)
        output = self.drop_out(output)
        output = F.relu(self.fc1(output))
        return F.log_softmax(self.fc2(output), dim=1)


def train(model, train_loader, optimizer):
    model.train()
    losses = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss = F.nll_loss(output, target)
        losses = losses + loss.data[0]

        loss.backward()
        optimizer.step()
    return model, (losses / data.size(0))


def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valid_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader.dataset)
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return correct / len(valid_loader.dataset), valid_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def experiment(model, epochs=10, lr=0.001, act_func, nb_layers):
    best_precision = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model, train_loss = train(model, train_loader, optimizer)
        precision, val_loss = valid(model, valid_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if precision > best_precision:
            best_precision = precision
            best_model = model

    fig = plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')

    plt.xlabel('Epoch')
    plt.xlim(0, epochs)

    plt.ylabel('Average negative log loss')
    plt.title('Average negative log loss ', type(model).__name__)

    plt.legend(loc='best')

    fig.savefig('./Graphs/Graph' + )
    #plt.show()

    return best_model, best_precision


best_precision = 0
for act_func in [F.sigmoid, F.tanh, F.relu]:
    for nb_layers in range(2, 5):
        for model in [FcNetwork(act_func, nb_layers), Cnn(act_func, nb_layers)]:  # add your models in the list
            model.cuda()  # if you have access to a gpu
            model, precision = experiment(model, act_func=act_func, nb_layers=nb_layers)
            if precision > best_precision:
                best_precision = precision
                best_model = model

test(best_model, test_loader)
