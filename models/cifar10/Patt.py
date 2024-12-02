import torch
from torch import nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class Model(nn.Module):
    def __init__(self, num_classes, image_size, dropout_rate=0.4, reduction_ratio=10):
        super(Model, self).__init__()
        self.input_shape = (3, image_size * image_size)
        self.num_classes = num_classes
        self.image_size = image_size

        # 创建三个分支，每个分支有不同的卷积核大小
        self.branch1 = self.create_branch(3, 32, 3, 0)
        self.branch2 = self.create_branch(32, 64, 4, 1)
        self.branch3 = self.create_branch(64, 128, 5, 1)

        # 创建通道注意力机制
        self.channel_attention = ChannelAttention(32)

        # 创建空间注意力机制
        self.spatial_attention = SpatialAttention()

        # 创建全连接层
        self.fc1 = nn.Linear(32 + 64 + 128, 2048)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(2048, num_classes)

    def create_branch(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        x = x.view(-1, 3, self.image_size, self.image_size)

        # 在每个分支上执行前向传播
        out1 = self.branch1(x)
        out2 = self.branch2(out1)
        out3 = self.branch3(out2)

        # 对较浅层次特征应用通道注意力机制
        out1 = self.channel_attention(out1)

        # 对较深层次特征应用空间注意力机制
        out3 = self.spatial_attention(out3)

        # 对每个分支的输出进行全局平均池化
        out1 = F.adaptive_avg_pool2d(out1, (1, 1))
        out2 = F.adaptive_avg_pool2d(out2, (1, 1))
        out3 = F.adaptive_avg_pool2d(out3, (1, 1))

        # 将三个分支的输出合并
        x = torch.cat([out1, out2, out3], dim=1)

        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x