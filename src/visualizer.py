"""
可视化模块 - 生成训练曲线和预测结果图
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_history(self, history, save_name='training_curves.png'):
        """
        绘制训练历史曲线
        - 训练/测试损失曲线
        - 测试准确率曲线
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 损失曲线
        ax1 = axes[0]
        ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', 
                 linewidth=2, markersize=6)
        ax1.plot(epochs, history['test_loss'], 'r-s', label='Test Loss', 
                 linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xticks(epochs)
        
        # 准确率曲线
        ax2 = axes[1]
        ax2.plot(epochs, history['test_accuracy'], 'g-^', label='Test Accuracy', 
                 linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Test Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xticks(epochs)
        
        # 标注最高准确率
        max_acc = max(history['test_accuracy'])
        max_epoch = history['test_accuracy'].index(max_acc) + 1
        ax2.annotate(f'Best: {max_acc:.2f}%', 
                     xy=(max_epoch, max_acc),
                     xytext=(max_epoch + 0.5, max_acc - 2),
                     fontsize=10, color='green',
                     arrowprops=dict(arrowstyle='->', color='green'))
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Training curves saved to: {save_path}")
        
    def plot_predictions(self, model, test_images, test_labels, device,
                        num_samples=25, save_name='predictions.png'):
        """
        绘制预测结果网格图
        - 绿色标题: 预测正确
        - 红色标题: 预测错误
        """
        model.eval()
        
        # 随机选择样本
        indices = np.random.choice(len(test_images), num_samples, replace=False)
        
        # 计算网格大小
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                image = test_images[idx:idx+1].to(device)
                label = test_labels[idx].item()
                
                # 预测
                output = model(image)
                _, predicted = output.max(1)
                pred = predicted.item()
                
                # 绘制图像
                ax = axes[i]
                ax.imshow(test_images[idx].squeeze(), cmap='gray')
                ax.axis('off')
                
                # 设置标题颜色
                if pred == label:
                    color = 'green'
                    title = f'Pred: {pred}'
                else:
                    color = 'red'
                    title = f'Pred: {pred} (True: {label})'
                    
                ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        
        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle('Model Predictions on Test Samples', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Predictions saved to: {save_path}")
        
    def plot_confusion_matrix(self, model, test_loader, device, 
                              save_name='confusion_matrix.png'):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to: {save_path}")