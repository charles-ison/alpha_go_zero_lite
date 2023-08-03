import torch.nn as nn


class TicTacToeResNet(nn.Module):
    def __init__(self):
        super(TicTacToeResNet, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=0, padding=0)
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=0, padding=0)
        self.relu = nn.ReLU()
        self.out = nn.Linear(3 * 3 * 3, 1)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
