import torch.nn as nn
import torch.nn.functional as F
from typing import cast


# Construct a model with one layer
class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()
        self.l1 = nn.Linear(4, 1)


    def forward(self, inputs):
        outputs = self.l1(inputs)
        return outputs


class net2_layers(nn.Module):

    def __init__(self, activation='relu', num_classes=10):
        super(TestNet, self).__init__()
        self.num_classes = num_classes
        self.l1 = nn.Linear(16 * 16, 100)
        self.l2 = nn.Linear(100, self.num_classes)
        self.flat = nn.Flatten()
        # Input size is 16*16, output size should be the same with the number of classes

    def forward(self, inputs):
        inputs = self.flat(inputs)
        h = F.relu(self.l1(inputs))
        outputs = F.softmax(self.l2(h), dim=1)  # Use softmax as the activation function for the last layer
        return outputs


def build_config(activations, conv_layers, lin_layers, dropout):
    return dict(activations=activations, conv_layers=conv_layers, linear_layers=lin_layers, dropout=dropout)


class Lin1_5_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 5)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(5,1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return self.flatten(x)

class Lin1_10_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 10)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(10, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return self.flatten(x)

class Lin1_15_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 15)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(15, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return self.flatten(x)

class Lin1_20_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 20)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(20, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return self.flatten(x)

class Lin1_30_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 30)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(30, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return self.flatten(x)

class Lin2_5_5_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 5)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(5, 5)
        self.act2 = nn.LeakyReLU()
        self.lin3 = nn.Linear(5, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        return self.flatten(x)

class Lin2_10_10_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 10)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(10, 10)
        self.act2 = nn.LeakyReLU()
        self.lin3 = nn.Linear(10, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        return self.flatten(x)

class Lin2_10_5_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 10)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(10, 5)
        self.act2 = nn.LeakyReLU()
        self.lin3 = nn.Linear(5, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        return self.flatten(x)

class Lin2_20_10_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 20)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(20, 10)
        self.act2 = nn.LeakyReLU()
        self.lin3 = nn.Linear(10, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        return self.flatten(x)

class Lin2_20_5_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 20)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(20, 5)
        self.act2 = nn.LeakyReLU()
        self.lin3 = nn.Linear(5, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        return self.flatten(x)

class Lin3_10_5_5_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 10)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(10, 5)
        self.act2 = nn.LeakyReLU()
        self.lin3 = nn.Linear(5, 5)
        self.act3 = nn.LeakyReLU()
        self.lin4 = nn.Linear(5, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        x = self.act3(x)
        x = self.lin4(x)
        return self.flatten(x)

class Lin3_10_10_5_Net(nn.Module):

    def __init__(self, activations=None):
        super(Lin1_5_Net, self).__init__()
        self.lin1 = nn.Linear(4, 20)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(20, 5)
        self.act2 = nn.LeakyReLU()
        self.lin3 = nn.Linear(5, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        return self.flatten(x)


class Net(nn.Module):

    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        out_channels = 4
        self.act = config["activations"]
        self.regressor = self.make_linear_layers(out_channels) #remplace self.classifier
        #self.softmax = nn.Softmax(dim=1)

    def make_linear_layers(self, in_channels):
        layers = []
        nb_layers = 0
        act = self.config['activations']
        for v in self.config['linear_layers']:
            v = cast(int, v)
            linear = nn.Linear(in_channels, v)
            in_channels = v
            if self.config['dropout'] != 0 and nb_layers == 1:
                layers.append(nn.Dropout(self.config['dropout']))

            if act == 'relu':
                layers.append(linear, nn.ReLU())
            elif act == 'tanh':
                layers.append(linear, nn.Tanh())
            elif act == 'sigmoid':
                layers.append(linear, nn.Sigmoid())
            elif act == 'prelu':
                layers.append(linear, nn.PReLU())
            nb_layers += 1

        layers.append(nn.Linear(in_channels, 1))
        nb_layers += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.regressor(x)


