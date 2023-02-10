import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        return x.squeeze(-1).squeeze(-1)
