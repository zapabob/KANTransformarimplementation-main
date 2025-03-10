"""
MNISTデータセットを使用した実験クラス
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

class MNISTExperiment(BaseExperiment):
    """MNISTデータセットでの実験クラス"""
    
    def __init__(self, config: BioKANConfig):
        super().__init__(config, "mnist_experiment")
        
        self.prepare_data()
        self.create_model()
        
        # 損失関数と最適化器の設定
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
    def prepare_data(self):
        """MNISTデータの準備"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 訓練データ
        self.train_dataset = datasets.MNIST(
            'data', train=True, download=True,
            transform=transform
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # テストデータ
        self.test_dataset = datasets.MNIST(
            'data', train=False,
            transform=transform
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size
        )
        
    def create_model(self):
        """モデルの作成"""
        self.model = NeuropharmacologicalBioKAN(self.config)
        self.model = self.model.to(self.device)
        
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
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 精度の計算
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
        
        return metrics
        
    def validate(self) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
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
                
        metrics = {
            'loss': total_loss / len(self.test_loader),
            'accuracy': 100. * correct / total
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
            self.log_metrics(val_metrics, "val")
            
            # モデルの保存
            is_best = val_metrics['loss'] < self.best_metric
            if is_best:
                self.best_metric = val_metrics['loss']
            self.save_checkpoint(is_best)
            
        # 最終結果の保存
        self.save_results() 