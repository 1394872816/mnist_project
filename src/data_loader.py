"""
数据加载模块 - 负责加载和预处理MNIST数据
"""
import gzip
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


class MNISTLoader:
    """MNIST数据加载器"""
    
    def __init__(self, data_dir, batch_size=128):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
    def _load_images(self, filename):
        """加载图像数据"""
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            # 读取文件头信息
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            # 读取图像数据
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, 1, rows, cols)
        return data.astype(np.float32) / 255.0  # 归一化到[0,1]
    
    def _load_labels(self, filename):
        """加载标签数据"""
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            # 读取文件头信息
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            # 读取标签数据
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data
    
    def get_data_loaders(self):
        """获取训练和测试数据加载器"""
        # 加载训练数据
        train_images = self._load_images('train-images-idx3-ubyte.gz')
        train_labels = self._load_labels('train-labels-idx1-ubyte.gz')
        
        # 加载测试数据
        test_images = self._load_images('t10k-images-idx3-ubyte.gz')
        test_labels = self._load_labels('t10k-labels-idx1-ubyte.gz')
        
        # 转换为PyTorch张量
        train_images = torch.FloatTensor(train_images)
        train_labels = torch.LongTensor(train_labels)
        test_images = torch.FloatTensor(test_images)
        test_labels = torch.LongTensor(test_labels)
        
        # 创建数据集
        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"图像尺寸: {train_images.shape[1:]} (C×H×W)")
        
        return train_loader, test_loader, test_images, test_labels