import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size=100, d=64):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(input_size, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def forward(self, input):
        input = input.unsqueeze(2).unsqueeze(3)
        # [N, 100, 1, 1]
        output = F.relu(self.deconv1_bn(self.deconv1(input)))   # [N, d*8, 4, 4]
        output = F.relu(self.deconv2_bn(self.deconv2(output)))  # [N, d*4, 8, 8]
        output = F.relu(self.deconv3_bn(self.deconv3(output)))  # [N, d*2, 16, 16]
        output = F.relu(self.deconv4_bn(self.deconv4(output)))  # [N, d, 32, 32]
        output = F.tanh(self.deconv5(output))                   # [N, 1, 64, 64]

        return output


class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    def forward(self, input):
        # [N, 1, 64, 64]
        input = input.view(input.size(0), 1, 64, 64)
        output = F.leaky_relu(self.conv1(input), 0.2)                   # [N, d, 32, 32]
        output = F.leaky_relu(self.conv2_bn(self.conv2(output)), 0.2)   # [N, d*2, 16, 16]
        output = F.leaky_relu(self.conv3_bn(self.conv3(output)), 0.2)   # [N, d*4, 8, 8]
        output = F.leaky_relu(self.conv4_bn(self.conv4(output)), 0.2)   # [N, d*8, 4, 4]
        output = F.sigmoid(self.conv5(output))                          # [N, 1, 1, 1]

        return output

