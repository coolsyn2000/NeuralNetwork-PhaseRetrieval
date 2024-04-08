import torch
import torch.nn as nn

class Contracting(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1),
                                    nn.Conv2d(32, 32, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(0.2))

        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=1, padding=1),
                                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2))

        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2))

        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                    nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2))

        self.layer5 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2))

        self.down_sample = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x1 = self.layer1(x)  # (128,128,32)
        x2 = self.layer2(self.down_sample(x1))  # (64,64,64)
        x3 = self.layer3(self.down_sample(x2))  # (32,32,128)
        x4 = self.layer4(self.down_sample(x3))  # (16,16,256)
        # return X4,X3,X2,X1
        x5 = self.layer5(self.down_sample(x4))  # (8,8,512)
        return x5, x4, x3, x2, x1


class Expansive(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(32, 1, 1, stride=1, padding=0))

        self.layer2 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(0.2))

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2))

        self.layer4 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2))

        self.layer5 = nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2))

        self.up_sample_54 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.up_sample_43 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.up_sample_32 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up_sample_21 = nn.ConvTranspose2d(64, 32, 2, stride=2)

    def forward(self, x5, x4, x3, x2, x1):
        x = self.up_sample_54(x5)  # (16,16,256)
        x4 = torch.cat([x, x4], dim=1)  # (16,16,512)
        x4 = self.layer5(x4)  # (16,16,256)

        x = self.up_sample_43(x4)  # (32,32,128)
        x3 = torch.cat([x, x3], dim=1)  # (32,32,256)
        x3 = self.layer4(x3)  # (32,32,128)

        x = self.up_sample_32(x3)  # (64,64,64)
        x2 = torch.cat([x, x2], dim=1)  # (64,64,128)
        x2 = self.layer3(x2)  # (64,64,64)

        x = self.up_sample_21(x2)  # (128,128,32)
        x1 = torch.cat([x, x1], dim=1)  # (128,128,64)
        x1 = self.layer2(x1)  # (128,128,32)

        x = self.layer1(x1)  # (128,128,1)

        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = Contracting()
        self.up = Expansive()

    def forward(self, X):
        x5, x4, x3, x2, x1 = self.down(X)
        x = self.up(x5, x4, x3, x2, x1)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, stride, 1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.conv(x)


