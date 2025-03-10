"""
BioKANモデルのトレーニングクラス
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import numpy as np
from tqdm import tqdm

from biokan.core.config import BioKANConfig


class BioKANTrainer:
    """BioKANモデルのトレーニングを管理するクラス"""
    
    def __init__(self, model: nn.Module, config: BioKANConfig):
        """
        Args:
            model: 訓練対象のBioKANモデル
            config: トレーニング設定
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # 訓練履歴
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_model_state = None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        pbar: Optional[tqdm] = None
    ) -> Tuple[float, float]:
        """1エポックの訓練を実行

        Args:
            train_loader: 訓練データローダー
            epoch: 現在のエポック数
            pbar: プログレスバー（オプション）

        Returns:
            (平均損失, 精度)のタプル
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = data.view(data.size(0), -1)  # フラット化
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if pbar:
                pbar.set_postfix({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'acc': 100. * correct / total
                })
                pbar.update(1)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def evaluate(
        self,
        val_loader: DataLoader,
        pbar: Optional[tqdm] = None
    ) -> Tuple[float, float]:
        """モデルの評価を実行

        Args:
            val_loader: 検証データローダー
            pbar: プログレスバー（オプション）

        Returns:
            (平均損失, 精度)のタプル
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)  # フラット化
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if pbar:
                    pbar.update(1)
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        return val_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> None:
        """モデルの訓練を実行

        Args:
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            num_epochs: エポック数（指定がない場合はconfigの値を使用）
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        total_steps = num_epochs * (len(train_loader) + len(val_loader))
        
        with tqdm(total=total_steps, desc="Training") as pbar:
            for epoch in range(1, num_epochs + 1):
                # 訓練
                train_loss, train_acc = self.train_epoch(
                    train_loader, epoch, pbar)
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                
                # 評価
                val_loss, val_acc = self.evaluate(val_loader, pbar)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # 最良モデルの保存
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_model_state = self.model.state_dict().copy()
                
                # 進捗の表示
                pbar.set_postfix({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_acc': self.best_val_accuracy
                })
    
    def get_best_model(self) -> nn.Module:
        """最良の検証精度を達成したモデルを返す"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.model
    
    def get_training_history(self) -> dict:
        """訓練の履歴を返す"""
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy
        } 