import torch
import torch.nn as nn



class Contracting(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, 3, stride=1, padding=1),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=1, padding=1),
                                    nn.ReLU())

        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=1, padding=1),
                                    nn.ReLU())

        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                    nn.ReLU())

        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                    nn.ReLU())

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

        self.layer1 = nn.Sequential(nn.Conv2d(16, 1, 1, stride=1, padding=0),
                                    nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(32, 16, 3, stride=1, padding=1),
                                    nn.ReLU())

        self.layer3 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1),
                                    nn.ReLU())

        self.layer4 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1),
                                    nn.ReLU())

        self.layer5 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1),
                                    nn.ReLU())

        self.up_sample_54 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.up_sample_43 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up_sample_32 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.up_sample_21 = nn.ConvTranspose2d(32, 16, 2, stride=2)

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

        return x*torch.pi


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = Contracting()
        self.up = Expansive()

    def forward(self, X):
        x5, x4, x3, x2, x1 = self.down(X)
        x = self.up(x5, x4, x3, x2, x1)
        return x

class SIREN(nn.Module):
    def __init__(self, omega_0=30.0):
        super(SIREN, self).__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


class MLPToCNN(nn.Module):
    def __init__(self):
        super(MLPToCNN, self).__init__()

        # 定义三层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),  # 输出为 32 * 32
            nn.Tanh(),
            nn.Linear(512, 1024),  # 输出为 32 * 32
            nn.Tanh(),
        )

        # 定义 CNN 解码器，直接扩张，保持通道数为1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),  # (1, 32, 32) -> (64, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  # (1, 32, 32) -> (64, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (128, 128, 128) -> (256, 256, 256)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # (256, 256, 256) -> (1, 512, 512)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mlp(x)  # 经过 MLP
        x = x.view(-1, 1, 32, 32)  # 重塑为 (batch_size, 1, 32, 32)
        x = self.decoder(x)  # 经过 CNN 解码器
        return x*torch.pi