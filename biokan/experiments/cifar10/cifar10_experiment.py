"""
CIFAR-10データセットを使用した実験クラス
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, Tuple

from ..base_experiment import BaseExperiment
from ...core.config import BioKANConfig
from ...core.biokan_model import NeuropharmacologicalBioKAN

class CIFAR10Experiment(BaseExperiment):
    """CIFAR-10データセットでの実験クラス"""
    
    def __init__(self, config: BioKANConfig):
        super().__init__(config, "cifar10_experiment")
        
        # クラスラベル
        self.classes = ['airplane', 'automobile', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.prepare_data()
        self.create_model()
        
        # 損失関数と最適化器の設定
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # データ拡張の設定
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                              (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                              (0.2023, 0.1994, 0.2010))
        ])
        
    def prepare_data(self):
        """CIFAR-10データの準備"""
        # 訓練データ
        self.train_dataset = datasets.CIFAR10(
            'data', train=True, download=True,
            transform=self.train_transform
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # テストデータ
        self.test_dataset = datasets.CIFAR10(
            'data', train=False,
            transform=self.test_transform
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            pin_memory=True
        )
        
    def create_model(self):
        """モデルの作成"""
        self.model = NeuropharmacologicalBioKAN(self.config)
        self.model = self.model.to(self.device)
        
        # 学習率スケジューラの設定
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs
        )
        
    def train_epoch(self) -> Dict[str, float]:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 精度の計算
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        # 学習率の更新
        self.scheduler.step()
        
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return metrics
        
    def validate(self) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # 精度の計算
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # クラスごとの精度を計算
                correct_tensor = pred.eq(target.view_as(pred))
                for i in range(len(target)):
                    label = target[i].item()
                    class_correct[label] += correct_tensor[i].item()
                    class_total[label] += 1
                    
        # クラスごとの精度を計算
        class_accuracy = {}
        for i in range(10):
            if class_total[i] > 0:
                class_accuracy[self.classes[i]] = 100. * class_correct[i] / class_total[i]
                
        metrics = {
            'loss': total_loss / len(self.test_loader),
            'accuracy': 100. * correct / total,
            'class_accuracy': class_accuracy
        }
        
        return metrics
        
    def run(self, num_epochs: int):
        """実験の実行"""
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 訓練
            train_metrics = self.train_epoch()
            self.log_metrics(train_metrics, "train")
            
            # 検証
            val_metrics = self.validate()
            self.log_metrics(
                {k: v for k, v in val_metrics.items() 
                 if not isinstance(v, dict)},
                "val"
            )
            
            # クラスごとの精度をロギング
            for class_name, accuracy in val_metrics['class_accuracy'].items():
                self.log_metrics(
                    {f"class_{class_name}_accuracy": accuracy},
                    "val"
                )
            
            # モデルの保存
            is_best = val_metrics['loss'] < self.best_metric
            if is_best:
                self.best_metric = val_metrics['loss']
            self.save_checkpoint(is_best)
            
        # 最終結果の保存
        self.save_results() 