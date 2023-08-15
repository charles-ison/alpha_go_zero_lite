import torch.nn as nn


class TicTacToeCNN(nn.Module):
    def __init__(self):
        super(TicTacToeCNN, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1)
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
        self.conv0 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(18 * 3 * 3, 3 * 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x.view(-1, 3, 3)


class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(18 * 3 * 3, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return self.tanh(x)
