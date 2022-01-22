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


def build_config(activations, conv_layers, lin_layers):
    return dict(activations=activations, conv_layers=conv_layers, linear_layers=lin_layers)


class Net(nn.Module):

    def __init__(self, config, num_classes=10):
        super(Net, self).__init__()
        self.num_classes = num_classes

        if config["conv_layers"] != 0:
            self.features, out_feat_channels = self.make_conv_layers(config)
            out_channels = out_feat_channels * (16/(2*config["conv_layers"].count('M'))) * (16/(2*config["conv_layers"].count('M')))
        else:
            self.features, out_channels = None, 256
        self.act = config["activations"]
        self.flatten = nn.Flatten()
        self.classifier = self.make_linear_layers(config, out_channels)
        self.softmax = nn.Softmax(dim=1)


    def make_conv_layers(self, config):
        """
        :param config: List[Union[str, int]]
        :return: the model's architecture
        """
        layers = []
        in_channels = 1
        act = config['activations']
        for v in config['conv_layers']:
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

    def make_linear_layers(self, config, in_channels):
        layers = []
        act = config['activations']
        for v in config['linear_layers']:
            v = cast(int, v)
            linear = nn.Linear(in_channels, v)
            in_channels = v
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
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.features is not None:
            x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


