import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder    = Encoder()
        self.decoder    = Decoder()
        self.classifier = Classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1      = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.act1       = nn.ReLU()
        self.pool1      = nn.MaxPool2d(2)

        self.conv2      = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.act2       = nn.ReLU()
        self.pool2      = nn.MaxPool2d(2)  

        self.fc3        = nn.Linear(32*7*7, 64)
        self.act3       = nn.ReLU()
        self.dropout3   = nn.Dropout(0.3)

        self.fc4        = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(-1, 32*7*7)

        x = self.fc3(x)
        x = self.act3(x)
        # x = self.dropout3(x)

        x = self.fc4(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1        = nn.Linear(2, 64)
        self.act1       = nn.ReLU()

        self.fc2        = nn.Linear(64, 32*7*7)
        self.act2       = nn.ReLU()

        self.upsample3  = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3      = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.act3       = nn.ReLU()

        self.upsample4  = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv4      = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.act4       = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = x.view(-1, 32, 7, 7)

        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)

        x = self.upsample4(x)
        x = self.conv4(x)
        x = self.act4(x)

        return x    


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)