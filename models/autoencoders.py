import torch as ch
import numpy as np
import torch.nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as snorm
from torch import nn


import torch
import torch.nn as nn

class CAE3(nn.Module):
    def __init__(self, latent_dim=512):
        super(CAE3, self).__init__()

        self.encoder = Encoder3(latent_dim)
        self.decoder = Decoder3(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Encoder3(nn.Module):
    def __init__(self, latent_dim=512):
        super(Encoder3, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder3(nn.Module):
    def __init__(self, latent_dim=512):
        super(Decoder3, self).__init__()

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.conv_transpose1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv_transpose3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.conv_transpose4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.conv_transpose1(x)
        x = self.relu1(x)
        x = self.conv_transpose2(x)
        x = self.relu2(x)
        x = self.conv_transpose3(x)
        x = self.relu3(x)
        x = self.conv_transpose4(x)
        return x


class CAE2(nn.Module):
    def __init__(self, latent_dim=512):
        super(CAE2, self).__init__()

        self.encoder = Encoder2(latent_dim)
        self.decoder = Decoder2(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Encoder2(nn.Module):
    def __init__(self, latent_dim=512):
        super(Encoder2, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(128 * 3 * 3, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder2(nn.Module):
    def __init__(self, latent_dim=512):
        super(Decoder2, self).__init__()

        self.fc = nn.Linear(latent_dim, 128 * 3 * 3)
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, output_padding=0)
        self.relu1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=0)
        self.relu2 = nn.ReLU()
        self.conv_transpose3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=0, output_padding=1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 3, 3)
        x = self.conv_transpose1(x)
        x = self.relu1(x)

        x = self.conv_transpose2(x)
        x = self.relu2(x)
        x = self.conv_transpose3(x)
        #x = self.sigmoid(x)
        return x

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(128 * 3 * 3, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 128 * 3 * 3)
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, output_padding=0)
        self.relu1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ReLU()
        self.conv_transpose3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 3, 3)
        x = self.conv_transpose1(x)
        x = self.relu1(x)
        x = self.conv_transpose2(x)
        x = self.relu2(x)
        x = self.conv_transpose3(x)
        x = self.sigmoid(x)

        return x


class CAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(CAE, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        #print(decoded.)
        return decoded

