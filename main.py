"""
MNIST手写数字识别 - 主程序入口
带有进度条和GPU实时监控功能
"""
# ============================================
#               环境变量设置
# ============================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================
#               标准库导入
# ============================================
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================
#               自定义模块导入
# ============================================
from config import Config
from src.data_loader import MNISTLoader
from src.model import MNISTNet, count_parameters
from src.trainer import Trainer
from src.visualizer import Visualizer
from src.device_monitor import DeviceMonitor


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_banner():
    """打印程序横幅"""
    banner = """
    ============================================================
    |                                                          |
    |           MNIST Handwritten Digit Recognition            |
    |                                                          |
    |              CNN-based Classification                    |
    |                                                          |
    ============================================================
    """
    print(banner)


def main():
    # 打印横幅
    print_banner()
    
    # 设置随机种子
    set_seed(Config.SEED)
    
    # 创建输出目录
    Config.create_dirs()
    
    # ==================== 0. 设备检测 ====================
    print("\n" + "=" * 60)
    print("Step [0/4] Device Detection")
    print("=" * 60)
    
    device_monitor = DeviceMonitor()
    device_monitor.print_device_info()
    
    # ==================== 1. 加载数据 ====================
    print("\n" + "=" * 60)
    print("Step [1/4] Loading Data")
    print("=" * 60)
    
    data_loader = MNISTLoader(Config.DATA_DIR, Config.BATCH_SIZE)
    train_loader, test_loader, test_images, test_labels = data_loader.get_data_loaders()
    
    print(f"\nData loading complete!")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Testing batches : {len(test_loader)}")
    
    # ==================== 2. 创建模型 ====================
    print("\n" + "=" * 60)
    print("Step [2/4] Creating Model")
    print("=" * 60)
    
    model = MNISTNet(
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE
    )
    
    print(f"\nModel Information:")
    print(f"  - Parameters : {count_parameters(model):,}")
    print(f"  - Dropout    : {Config.DROPOUT_RATE}")
    print(f"  - LR         : {Config.LEARNING_RATE}")
    print(f"  - Batch Size : {Config.BATCH_SIZE}")
    print(f"  - Epochs     : {Config.EPOCHS}")
    
    # ==================== 3. 训练模型 ====================
    print("\n" + "=" * 60)
    print("Step [3/4] Training Model")
    print("=" * 60)
    
    trainer = Trainer(
        model, 
        Config.DEVICE, 
        Config.LEARNING_RATE,
        device_monitor=device_monitor
    )
    history = trainer.train(train_loader, test_loader, Config.EPOCHS)
    
    # 保存模型
    model_path = f"{Config.MODEL_DIR}/mnist_cnn.pth"
    trainer.save_model(model_path)
    
    # ==================== 4. 可视化结果 ====================
    print("\n" + "=" * 60)
    print("Step [4/4] Generating Visualizations")
    print("=" * 60)
    
    visualizer = Visualizer(Config.FIGURE_DIR)
    
    # 绘制训练曲线
    print("\nGenerating training curves...")
    visualizer.plot_training_history(history)
    
    # 绘制预测结果
    print("\nGenerating prediction results...")
    visualizer.plot_predictions(
        model, test_images, test_labels, 
        Config.DEVICE, num_samples=25
    )
    
    # 绘制混淆矩阵
    try:
        print("\nGenerating confusion matrix...")
        visualizer.plot_confusion_matrix(model, test_loader, Config.DEVICE)
    except ImportError as e:
        print(f"Skipping confusion matrix: {e}")
    
    # ==================== 完成 ====================
    print("\n" + "=" * 60)
    print("Project Complete!")
    print("=" * 60)
    
    # 清理资源
    device_monitor.cleanup()
    
    print(f"""
    Output Files:
      - Model          : {model_path}
      - Training Curves: {Config.FIGURE_DIR}/training_curves.png
      - Predictions    : {Config.FIGURE_DIR}/predictions.png
      - Confusion Matrix: {Config.FIGURE_DIR}/confusion_matrix.png
    """)


if __name__ == '__main__':
    main()