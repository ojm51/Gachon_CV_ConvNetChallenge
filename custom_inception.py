import torch.nn as nn
import torch
import torch.nn.functional as F


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()

        self.layers = nn.Sequential(
            # Convolutional 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Convolutional 2
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Convolutional 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # pooling 1
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Convolutional 4
            nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),

            # Convolutional 5
            nn.Conv2d(80, 192, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            # Convolutional 6
            nn.Conv2d(192, 288, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(288),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class InceptionBlock_A(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock_A, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0)

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)

        return out


class InceptionBlock_B(nn.Module):
    def __init__(self, in_channels, n=7):
        super(InceptionBlock_B, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=1, padding=(0, 2 // n)),
            nn.Conv2d(128, 192, kernel_size=(1, 1), stride=1, padding=(2 // n, 0)),
            nn.Conv2d(192, 192, kernel_size=(1, 1), stride=1, padding=(0, 2 // n)),
            nn.Conv2d(192, 320, kernel_size=(1, 1), stride=1, padding=(2 // n, 0)),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 192, kernel_size=(1, 1), stride=1, padding=(0, 2 // n)),
            nn.Conv2d(192, 320, kernel_size=(1, 1), stride=1, padding=(2 // n, 0))
        )

        self.branch3 = nn.Conv2d(in_channels, 320, kernel_size=1, stride=1, padding=0)

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_channels, 320, kernel_size=1, stride=1, padding=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)

        return out


class InceptionBlock_C(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock_C, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.subbranch1_1 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.subbranch1_2 = nn.Conv2d(128, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))

        self.branch2 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.subbranch2_1 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.subbranch2_2 = nn.Conv2d(128, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))

        self.branch3 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=0)

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=1)
        )

    def forward(self, x):
        out_1 = self.branch1(x)
        out_1_1 = self.subbranch1_1(out_1)
        out_1_2 = self.subbranch1_2(out_1)
        out_2 = self.branch2(x)
        out_2_1 = self.subbranch2_1(out_2)
        out_2_2 = self.subbranch2_2(out_2)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        x = torch.cat([out_1_1, out_1_2, out_2_1, out_2_2, out3, out4], dim=1)

        return x


class inception_net(nn.Module):
    def __init__(self):
        super(inception_net, self).__init__()

        self.front = StemBlock()
        self.middle_A = InceptionBlock_A(288)
        self.middle_B = InceptionBlock_B(768)
        self.middle_C = InceptionBlock_C(1280)
        self.back = nn.MaxPool2d(kernel_size=24)
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.front(x)
        x = self.middle_A(x)
        x = self.middle_B(x)
        x = self.middle_C(x)
        x = self.back(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

