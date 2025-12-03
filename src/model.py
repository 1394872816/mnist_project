"""
CNN模型定义 - MNIST手写数字识别网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块：Conv -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class MNISTNet(nn.Module):
    """
    MNIST手写数字识别CNN网络
    
    网络结构:
    - 3个卷积块，逐步提取特征
    - 2个全连接层用于分类
    - Dropout防止过拟合
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()
        
        # 卷积层块1: 1 -> 32 channels
        self.conv_block1 = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        )
        
        # 卷积层块2: 32 -> 64 channels
        self.conv_block2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        )
        
        # 卷积层块3: 64 -> 128 channels
        self.conv_block3 = nn.Sequential(
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2)  # 7x7 -> 3x3
        )
        
        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """使用He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """获取特征图（用于可视化）"""
        features = []
        x = self.conv_block1(x)
        features.append(x)
        x = self.conv_block2(x)
        features.append(x)
        x = self.conv_block3(x)
        features.append(x)
        return features


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)