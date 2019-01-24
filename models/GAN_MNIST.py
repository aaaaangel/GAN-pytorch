import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size=100, output_size=784):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, output_size)

    def forward(self, input):
        output = F.leaky_relu(self.fc1(input), 0.2)
        output = F.leaky_relu(self.fc2(output), 0.2)
        output = F.leaky_relu(self.fc3(output), 0.2)
        output = F.tanh(self.fc4(output))

        return output


class Discriminator(nn.Module):
    def __init__(self, input_size=784, output_size=1):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, input):
        output = F.leaky_relu(self.fc1(input), 0.2)
        output = F.dropout(output, 0.3)
        output = F.leaky_relu(self.fc2(output), 0.2)
        output = F.dropout(output, 0.3)
        output = F.leaky_relu(self.fc3(output), 0.2)
        output = F.dropout(output, 0.3)
        output = F.sigmoid(self.fc4(output))

        return output

