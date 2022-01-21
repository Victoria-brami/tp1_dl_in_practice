import torch.nn as nn



class ConvModel(nn.Module):

    def __init__(self, activation='relu', num_classes=10):
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 3, 1, 1)
        self.conv2 = nn.Conv2d(1, 3, 1, 1)
        self.conv3 = nn.Conv2d(1, 3, 1, 1)
        self.conv4 = nn.Conv2d(1, 3, 1, 1)
        self.conv5 = nn.Conv2d(1, 3, 1, 1)
        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear(128, self.num_classes)
        if activation == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
            self.act4 = nn.ReLU()
            self.act5 = nn.ReLU()
        elif activation == 'prelu':
            self.act1 = nn.PReLU()
            self.act2 = nn.PReLU()
            self.act3 = nn.PReLU()
            self.act4 = nn.PReLU()
            self.act5 = nn.PReLU()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.mp1(x)
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.mp2(x)
        x = x.flatten()
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes, output_dims):
        super().__init__()
        layers = []

        input_dim: int = 28 * 28
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return nn.Softmax(x)
