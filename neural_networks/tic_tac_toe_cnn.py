import torch.nn as nn


class TicTacToeCNN(nn.Module):
    def __init__(self):
        super(TicTacToeCNN, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        return self.policy_head(x), self.value_head(x)


class PolicyHead(nn.Module):
    def __init__(self):
        super(PolicyHead, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 2 * 2, 3 * 3)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = x.flatten()
        x = self.fc(x)
        return x.reshape(3, 3)


class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 3 * 3, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = x.flatten()
        return self.fc(x)
