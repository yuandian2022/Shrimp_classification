import torch
import torch.nn as nn

class Cmp_CNN(nn.Module):
    def __init__(self):
        super(Cmp_CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=23, stride=1),
            nn.LayerNorm([32, 2026]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 11, 1),
            nn.LayerNorm([64, 395]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 5, 1),
            nn.LayerNorm([128, 75]),
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 128, 5, 1),
            nn.LayerNorm([128, 11]),
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.dropout(x)
        output = self.linear(x)
        output = self.softmax(output)
        return output

class Cmp_CNN_draw(nn.Module):
    def __init__(self):
        super(Cmp_CNN_draw, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=20, stride=2)
        self.layer1 = nn.LayerNorm([32, 245])
        self.relu1 = nn.ReLU()
        self.maxpooling1 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv1d(32, 64, 20, 2)
        self.layer2 = nn.LayerNorm([64, 52])
        self.relu2 = nn.ReLU()
        self.maxpooling2 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv1d(64, 128, 20, 2)
        self.layer3 = nn.LayerNorm([128, 4])
        self.relu3 = nn.ReLU()
        self.maxpooling3 = nn.MaxPool1d(2, 1, ceil_mode=True)

        self.flatten = nn.Flatten()

    def forward(self, input):
        x = self.conv1(input)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.maxpooling2(x)
        x = self.conv3(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.maxpooling3(x)
        x = self.flatten(x)
        return x