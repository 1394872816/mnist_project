"""
配置文件 - 集中管理所有超参数
"""
import torch
import os


class Config:
    # 路径配置
    DATA_DIR = './dataset'
    OUTPUT_DIR = './outputs'
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
    
    # 训练超参数
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 15
    
    # 模型参数
    NUM_CLASSES = 10
    DROPOUT_RATE = 0.5
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机种子
    SEED = 42
    
    # 进度条配置
    SHOW_PROGRESS_BAR = True
    SHOW_GPU_MONITOR = True
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.FIGURE_DIR, exist_ok=True)