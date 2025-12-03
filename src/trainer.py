"""
训练器模块 - 负责模型的训练和评估（带进度条和GPU监控）
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, device, learning_rate=0.001, device_monitor=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )
        
        # 设备监控器
        self.device_monitor = device_monitor
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': []
        }
        
    def train_epoch(self, train_loader, epoch, total_epochs):
        """训练一个epoch（带进度条）"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 创建进度条 - 自动适应终端宽度
        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch:2d}/{total_epochs}',
            unit='batch',
            dynamic_ncols=True,  # 自动适应终端宽度
            leave=True
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 计算当前统计
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            
            # 更新进度条显示 - 简化信息避免截断
            if self.device_monitor and self.device_monitor.is_gpu:
                gpu_usage = self.device_monitor.get_gpu_usage()
                if gpu_usage and 'gpu_util' in gpu_usage:
                    pbar.set_postfix_str(
                        f"loss={avg_loss:.4f}, acc={acc:.2f}%, "
                        f"GPU={gpu_usage['memory_used']:.1f}GB({gpu_usage['gpu_util']}%)"
                    )
                else:
                    pbar.set_postfix_str(f"loss={avg_loss:.4f}, acc={acc:.2f}%")
            else:
                pbar.set_postfix_str(f"loss={avg_loss:.4f}, acc={acc:.2f}%")
            
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader, desc="Testing"):
        """评估模型（带进度条）"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # 创建进度条
        pbar = tqdm(
            test_loader,
            desc=desc,
            unit='batch',
            dynamic_ncols=True,
            leave=False
        )
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix_str(
                    f"loss={total_loss / (pbar.n + 1):.4f}, acc={100.0 * correct / total:.2f}%"
                )
                
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, test_loader, epochs):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"Start Training")
        print(f"{'='*60}")
        
        # 显示当前设备状态
        if self.device_monitor:
            print(f"Device Status: {self.device_monitor.format_usage_string()}")
        print()
        
        best_accuracy = 0
        
        for epoch in range(1, epochs + 1):
            # 显示当前epoch开始
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch}/{epochs}] Learning Rate: {current_lr:.6f}")
            print("-" * 60)
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch, epochs)
            
            # 评估
            test_loss, test_accuracy = self.evaluate(test_loader, desc="Evaluating")
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['test_accuracy'].append(test_accuracy)
            
            # 打印结果
            result_str = (f"Result: Train Loss={train_loss:.4f}, "
                         f"Test Loss={test_loss:.4f}, Test Acc={test_accuracy:.2f}%")
            
            # 检查是否是最佳模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                result_str += " [Best]"
                
            print(result_str)
                
            # 显示GPU使用情况
            if self.device_monitor:
                print(f"Resource: {self.device_monitor.format_usage_string()}")
                
        print(f"\n{'='*60}")
        print(f"Training Complete! Best Test Accuracy: {best_accuracy:.2f}%")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Model saved to: {path}")
        
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {path}")