import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(in_channels=74, out_channels=1024,
                                         kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)

        self.tconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=128,
                                         kernel_size=7, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                         kernel_size=4, stride=2, padding=1,
                                         bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1,
                                         kernel_size=4, stride=2, padding=1,
                                         bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))

        img = torch.sigmoid(self.tconv4(x))

        return img


class MNIST_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=1024,
                               kernel_size=7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        return x


class MNIST_DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1024, out_channels=1,
                              kernel_size=1)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))
        return output


class MNIST_QHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=128,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(in_channels=128, out_channels=10,
                                   kernel_size=1)

        self.conv_mu = nn.Conv2d(in_channels=128, out_channels=2,
                                 kernel_size=1)
        self.conv_var = nn.Conv2d(in_channels=128, out_channels=2,
                                  kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var

