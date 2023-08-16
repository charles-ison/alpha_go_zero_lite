import torch.nn as nn


class TicTacToeCNN(nn.Module):
    def __init__(self):
        super(TicTacToeCNN, self).__init__()
        num_channels = 128
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.policy_head = PolicyHead(num_channels)
        self.value_head = ValueHead(num_channels)

    def convolution_block(self, x):
        x = self.conv0(x)
        x = self.batch_norm(x)
        return self.relu(x)

    def residual_block0(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm(x)
        x += residual
        return self.relu(x)

    def residual_block1(self, x):
        residual = x
        x = self.conv3(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.batch_norm(x)
        x += residual
        return self.relu(x)

    def forward(self, x):
        x = self.convolution_block(x)
        x = self.residual_block0(x)
        x = self.residual_block1(x)
        return self.policy_head(x), self.value_head(x)


class PolicyHead(nn.Module):
    def __init__(self, num_channels):
        super(PolicyHead, self).__init__()
        num_in_channels = num_channels
        num_out_channels = 2 * num_channels
        self.conv0 = nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(num_out_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_out_channels * 3 * 3, 3 * 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x.view(-1, 3, 3)


class ValueHead(nn.Module):
    def __init__(self, num_channels):
        super(ValueHead, self).__init__()
        num_in_channels = num_channels
        num_out_channels = 2 * num_channels
        self.conv0 = nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(num_out_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_out_channels * 3 * 3, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv0(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return self.tanh(x)
