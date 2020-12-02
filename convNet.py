import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_channels = params.num_channels
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels * 6, kernel_size=3,
                               padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels * 6)
        self.conv2 = nn.Conv2d(in_channels=self.num_channels * 6, out_channels=self.num_channels * 12, kernel_size=3,
                               padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels * 12)
        self.conv3 = nn.Conv2d(in_channels=self.num_channels * 12, out_channels=self.num_channels * 24, kernel_size=3,
                               padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels * 24)

        self.fc1 = nn.Linear(in_features=self.num_channels * 24 * 4 * 4, out_features=120)
        self.fcbn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fcbn2 = nn.BatchNorm1d(60)
        self.out = nn.Linear(in_features=60, out_features=10)
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        s = self.bn1(self.conv1(s))  # batch_size x num_channels x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 16 x 16
        s = self.bn2(self.conv2(s))  # batch_size x num_channels x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 8 x 8
        s = self.bn3(self.conv3(s))  # batch_size x num_channels x 8 x 8
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 4 x 4

        # Flatten the output for each image
        s = s.reshape(-1, self.num_channels * 24 * 4 * 4)  # batch_size x num_channels x 4 x 4

        # Apply three fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  # batch_size x 120

        s = F.dropout(F.relu(self.fcbn2(self.fc2(s))),
                      p=self.dropout_rate, training=self.training)  # batch_size x 60

        s = self.out(s)  # batch_size x 10
        # s = F.softmax(s, dim=1)
        return s


def loss_fn(outputs, labels):
    return F.cross_entropy(outputs, labels)


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)
