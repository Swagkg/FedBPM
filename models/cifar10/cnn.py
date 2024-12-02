import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes, image_size):
        super(Model, self).__init__()
        self.input_shape = (1, image_size * image_size)
        self.num_classes = num_classes
        self.image_size = image_size

        # 创建卷积层和池化层，去掉self.conv4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # 创建全连接层

        self.fc1 = nn.Linear(128 * (image_size // 8) * (image_size // 8), 1024)  # 修改这里的输入维度
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = x.view(-1, 3, self.image_size, self.image_size)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


