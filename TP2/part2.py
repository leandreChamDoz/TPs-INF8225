from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from torchvision import transforms, datasets


def prepare_data():
    fashion = datasets.FashionMNIST

    train_data = fashion(
        "./data/fashion_mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    valid_data = fashion(
        "./data/fashion_mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)
    train_data.train_data = train_data.train_data[train_idx, :]
    train_data.train_labels = train_data.train_labels[
        torch.from_numpy(train_idx).type(torch.LongTensor)
    ]

    mask = np.ones(60000)
    mask[train_idx] = 0

    valid_data.train_data = valid_data.train_data[
        torch.from_numpy(np.argwhere(mask)), :
    ].squeeze()
    valid_data.train_labels = valid_data.train_labels[
        torch.from_numpy(mask).type(torch.ByteTensor)
    ]

    batch_size = 100
    test_batch_size = 100

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        fashion(
            "../data/fashion_mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
    )

    return train_loader, valid_loader, test_loader


plt.imshow(train_loader.dataset.train_data[1].numpy())

kernel_size = 5
kernel_size_pooling = 2


class FcNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        output = image.view(batch_size, -1)

        #output = F.sigmoid(self.fc1(output))
        output = F.relu(self.fc1(output))

        #output = F.sigmoid(self.fc2(output))
        output = F.relu(self.fc2(output))

        # output = F.log_softmax(self.fc2(output), dim=1)

        return F.log_softmax(self.fc3(output), dim=1)
        return output


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, image):
        output = self.kernel1(image)
        output = self.kernel2(output)
        output = output.reshape(output.size(0), -1)
        output = self.drop_out(output)

        #output = F.sigmoid(self.fc1(output))
        output = F.relu(self.fc1(output))

        #output = F.sigmoid(self.fc2(output))
        output = F.relu(self.fc2(output))

        #return F.log_softmax(self.fc2(output), dim=1)
        return F.log_softmax(self.fc3(output), dim=1)


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        losses += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = losses / len(train_loader.dataset)
    return model, average_loss


def valid(model, valid_loader):
    model.eval()
    total_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valid_loss += F.nll_loss(output, target).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader.dataset)
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    return correct.item() / len(valid_loader.dataset), valid_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(
        "test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def experiment(model, epochs=10, lr=0.001):
    best_precision = 0
    best_model = None
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    precisions = []
    for epoch in range(1, epochs + 1):
        model, train_loss = train(model, train_loader, optimizer)
        precision, val_loss = valid(model, valid_loader)
        precisions.append(precision * 100)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if precision > best_precision:
            best_precision = precision
            best_model = model

    return precisions,best_model, best_precision, losses_train, losses_validation

    plt.xlabel('Epoch')
    plt.xlim(0, epochs-1)

    plt.ylabel('Average negative log loss')
    plt.title('Average negative log loss ' + str(type(model).__name__))

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    fig.savefig('./Graphs/Graph_' + str(type(model).__name__) + '_ALL_' + '3_' + 'Relu')


    fig = plt.figure()
    plt.plot(precisions, label='validation')

    plt.xlabel('Epoch')
    plt.xlim(0, epochs-1)

    plt.ylabel('Precision (in %)')
    plt.title('Learning curve ' + str(type(model).__name__))

    plt.legend(loc='best')

    fig.savefig('./Graphs/Graph_' + str(type(model).__name__) + '_learning_' + '3_' + 'Relu')
    # plt.show()

    return best_model, best_precision

    best_precision = 0
    best_model = None

best_precision = 0
for model in [FcNetwork(), Cnn()]:  # add your models in the list
    model.cuda()  # if you have access to a gpu
    model, precision = experiment(model)
    if precision > best_precision:
        best_precision = precision
        best_model = model

    fig = plt.figure()
    fig.tight_layout(h_pad=3.2)
    # f, ax = plt.subplots(len(models))
    cols = 3
    precisions = []

    precisions, model, precision, losses_train, losses_validation = experiment(FcNetwork(), epochs=10)

    # for i, model in enumerate(models):
    #     model_name = model.__class__.__name__
    #     print("\nTraining model", model_name)
    #     precisions, model, precision, losses_train, losses_validation = experiment(model, epochs=10)
    #     model_stats.append((model_name, precision))
    #
    #     print(rows, cols)
    #     ax = fig.add_subplot(rows, cols, i + 1)
    #     ax.plot(losses_train, "-b", label="Training")
    #     ax.plot(losses_validation, "-r", label="Validation")
    #     ax.legend()
    #     ax.text(
    #         0.5,
    #         0.5,
    #         f"accuracy: {precision:.4f}",
    #         size=8,
    #         ha="center",
    #         transform=ax.transAxes,
    #     )
    #     ax.text(
    #         0.5,
    #         0.55,
    #         f"average loss: {sum(losses_validation)/ len(losses_validation):.4f}",
    #         size=8,
    #         ha="center",
    #         transform=ax.transAxes,
    #     )
    #     ax.set_title(f"{model_name}")
    #     ax.set_ylabel("Mean NLL")
    #     # ax.set_xlabel("Epoch")
    #
    #     if precision > best_precision:
    #         best_precision = precision
    #         best_model = model
    #
    # plt.show()
    #
    # print("model statistics", *model_stats, sep="\n\t- ")
    # print(
    #     "The best model is ",
    #     model.__class__.__name__,
    #     "with a validation precision of",
    #     best_precision,
    # )
    # print(model)
    test(best_model, test_loader)