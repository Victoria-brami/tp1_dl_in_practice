import torch.nn as nn
import torch.nn.functional as F
from typing import cast

class TestNet(nn.Module):

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


class Conv1Lin1Net(nn.Module):

    def __init__(self, activations=None, num_classes=10):
        super(Conv1Lin1Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.mp1 = nn.MaxPool2d(2)
        self.act1 = nn.LeakyReLU()
        self.lin1 = nn.Linear(512, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.mp1(x)
        x = self.flatten(x)
        x = self.lin1(x)
        return x


class Conv1Lin2Net(nn.Module):

    def __init__(self, activations=None, num_classes=10):
        super(Conv1Lin2Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.mp1 = nn.MaxPool2d(2)
        self.act1 = nn.LeakyReLU()
        self.lin1 = nn.Linear(512, 128)
        self.act2 = nn.LeakyReLU()
        self.lin2 = nn.Linear(128, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.mp1(x)
        x = self.flatten(x)
        x = self.act2(self.lin1(x))
        x = self.lin2(x)
        return x


class Conv2Lin1Net(nn.Module):

    def __init__(self, activations=None, num_classes=10):
        super(Conv2Lin1Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.mp1 = nn.MaxPool2d(2)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding='same')
        self.mp2 = nn.MaxPool2d(2)
        self.act2 = nn.LeakyReLU()
        self.lin1 = nn.Linear(256, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.mp1(x)
        x = self.act2(self.conv2(x))
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.lin1(x)
        return x


class Conv3Lin1Net(nn.Module):

    def __init__(self, activations=None, num_classes=10):
        super(Conv3Lin1Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.mp1 = nn.MaxPool2d(2)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding='same')
        self.mp2 = nn.MaxPool2d(2)
        self.act2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(16, 64, kernel_size=5, padding='same')
        self.mp3 = nn.MaxPool2d(2)
        self.act3 = nn.LeakyReLU()
        self.lin1 = nn.Linear(256, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.mp1(x)
        x = self.act2(self.conv2(x))
        x = self.mp2(x)
        x = self.act3(self.conv3(x))
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.lin1(x)
        return x


class Conv2Lin2Net(nn.Module):

    def __init__(self, activations=None, num_classes=10):
        super(Conv2Lin2Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.mp1 = nn.MaxPool2d(2)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding='same')
        self.mp2 = nn.MaxPool2d(2)
        self.act2 = nn.LeakyReLU()
        self.lin1 = nn.Linear(256, 128)
        self.act3 = nn.LeakyReLU()
        self.lin2 = nn.Linear(128, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.mp1(x)
        x = self.act2(self.conv2(x))
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.act3(self.lin1(x))
        x = self.lin2(x)
        return x



class Conv3Lin2Net(nn.Module):

    def __init__(self, activations=None, num_classes=10):
        super(Conv3Lin2Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.mp1 = nn.MaxPool2d(2)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding='same')
        self.mp2 = nn.MaxPool2d(2)
        self.act2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(16, 64, kernel_size=5, padding='same')
        self.mp3 = nn.MaxPool2d(2)
        self.act3 = nn.LeakyReLU()
        self.lin1 = nn.Linear(256, 128)
        self.act4 = nn.LeakyReLU()
        self.lin2 = nn.Linear(128, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.mp1(x)
        x = self.act2(self.conv2(x))
        x = self.mp2(x)
        x = self.act3(self.conv3(x))
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.act4(self.lin1(x))
        x = self.lin2(x)
        return x





class Net(nn.Module):

    def __init__(self, config, num_classes=10):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.config = config

        if config["conv_layers"] != 0:
            self.features, out_feat_channels = self.make_conv_layers(config)
            out_channels = out_feat_channels * (16/(2*config["conv_layers"].count('M'))) * (16/(2*config["conv_layers"].count('M')))
        else:
            self.features, out_channels = None, 256
        self.act = config["activations"]
        self.flatten = nn.Flatten()
        self.classifier = self.make_linear_layers(out_channels)
        self.softmax = nn.Softmax(dim=1)


    def make_conv_layers(self):
        """
        :param config: List[Union[str, int]]
        :return: the model's architecture
        """
        layers = []
        in_channels = 1
        act = self.config['activations']
        for v in self.config['conv_layers']:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                in_channels = v
                if act == 'relu':
                    layers += [conv2d, nn.ReLU()]
                elif act == 'tanh':
                    layers += [conv2d, nn.Tanh()]
                elif act == 'sigmoid':
                    layers += [conv2d, nn.Sigmoid()]
                elif act == 'prelu':
                    layers += [conv2d, nn.PReLU()]
        return nn.Sequential(*layers), in_channels

    def make_linear_layers(self, in_channels):
        layers = []
        nb_layers = 0
        act = self.config['activations']
        for v in self.config['linear_layers']:
            v = cast(int, v)
            linear = nn.Linear(in_channels, v)
            in_channels = v
            if self.config['dropout'] != 0 and nb_layers == 1:
                layers += [nn.Dropout(self.config['dropout'])]
            if v != self.num_classes:
                if act == 'relu':
                    layers += [linear, nn.ReLU()]
                elif act == 'tanh':
                    layers += [linear, nn.Tanh()]
                elif act == 'sigmoid':
                    layers += [linear, nn.Sigmoid()]
                elif act == 'prelu':
                    layers += [linear, nn.PReLU()]
            else:
                layers += [linear]
            nb_layers += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.features is not None:
            x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        # x = self.softmax(x)  # Not needed for CrossEntropy
        return x


