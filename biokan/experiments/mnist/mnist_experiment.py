"""
MNISTデータセットを使用した実験クラス
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, Tuple
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from ..base_experiment import BaseExperiment
from ...core.config import BioKANConfig
from ...core.biokan_model import NeuropharmacologicalBioKAN

class MNISTExperiment(BaseExperiment):
    """MNISTデータセットでの実験クラス"""
    
    def __init__(self, config: BioKANConfig):
        """
        MNISTの実験クラスの初期化
        
        Args:
            config (BioKANConfig): モデルの設定
        """
        super().__init__(config, "mnist_experiment")
        
        self.config = config
        self.device = config.device
        
        # データの準備
        self.prepare_data()
        
        # モデルの作成
        self.create_model()
        
    def prepare_data(self):
        """MNISTデータの準備"""
        # 訓練データの変換（データ拡張あり）
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation(10),  # データ拡張: 回転
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # データ拡張: 平行移動
            transforms.RandomErasing(p=0.2)  # データ拡張: ランダム消去
        ])
        
        # テストデータの変換（データ拡張なし）
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 訓練データ
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )
        
        # 訓練データと検証データに分割
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # テストデータ
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
    def create_model(self):
        """モデルの作成"""
        self.model = NeuropharmacologicalBioKAN(
            input_dim=784,  # MNIST画像サイズ (28x28)
            hidden_dim=self.config.hidden_dim,
            output_dim=10,  # MNISTのクラス数
            num_blocks=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            attention_type=self.config.attention_type,
            use_neuromodulation=self.config.use_neuromodulation
        )
        
        # モデルをGPUに移動
        self.model = self.model.to(self.device)
        
        # 損失関数とオプティマイザの設定
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay  # L2正則化
        )
        
        # 学習率スケジューラの設定
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,  # ウォームアップ期間
            div_factor=25,  # 初期学習率の除数
            final_div_factor=1e4  # 最終学習率の除数
        )
        
    def train_epoch(self) -> Dict[str, float]:
        """1エポックの訓練を実行"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # メモリ解放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # データをGPUに移動
                data, target = data.to(self.device), target.to(self.device)
                
                # データの形状を変更 [batch_size, 1, 28, 28] -> [batch_size, 784]
                batch_size = data.size(0)
                data = data.view(batch_size, -1)  # 784次元に平坦化
                
                # 勾配をゼロにリセット
                self.optimizer.zero_grad(set_to_none=True)  # メモリ効率を改善
                
                try:
                    # 順伝播
                    output = self.model(data)  # [batch_size, 10]
                    
                    # 損失の計算（ラベルスムージングを適用）
                    loss = self.label_smoothing_loss(output, target, smoothing=0.1)
                    
                    # 逆伝播
                    loss.backward()
                    
                    # 勾配クリッピング
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # パラメータの更新
                    self.optimizer.step()
                    
                    # 学習率の更新
                    self.scheduler.step()
                    
                    # 統計の更新
                    total_loss += loss.item()
                    pred = output.argmax(dim=1)  # [batch_size]
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                    
                    # メモリ解放
                    del output, loss, pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    print(f"バッチ処理中にエラーが発生しました: {e}")
                    continue
                
                # 進捗の表示
                if batch_idx % 10 == 0:
                    print(f'エポック: {self.current_epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                          f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                          f'損失: {loss.item():.6f}\t'
                          f'精度: {100. * correct / total:.2f}%')
                    
                    if wandb.run is not None:
                        wandb.log({
                            "batch_loss": loss.item(),
                            "batch_accuracy": 100. * correct / total,
                            "learning_rate": self.scheduler.get_last_lr()[0]
                        })
            
            # エポックの統計を返す
            return {
                'loss': total_loss / len(self.train_loader),
                'accuracy': 100. * correct / total
            }
            
        except Exception as e:
            print(f"訓練エポック中にエラーが発生しました: {e}")
            raise
    
    def label_smoothing_loss(self, pred, target, smoothing=0.1):
        """ラベルスムージングを適用した損失関数"""
        n_classes = pred.size(1)
        
        # one-hotエンコーディング
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        
        # スムージング
        smooth_one_hot = one_hot * (1 - smoothing) + smoothing / n_classes
        
        # 損失の計算
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        
        return loss
        
    def validate(self) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:  # 検証データを使用
                data, target = data.to(self.device), target.to(self.device)
                
                # データの形状を変更 [batch_size, 1, 28, 28] -> [batch_size, 784]
                batch_size = data.size(0)
                data = data.view(batch_size, -1)  # 784次元に平坦化
                
                output = self.model(data)  # [batch_size, 10]
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # 精度の計算
                pred = output.argmax(dim=1)  # [batch_size]
                correct += (pred == target).sum().item()
                total += target.size(0)
                
            metrics = {
                'loss': total_loss / len(self.val_loader),
                'accuracy': 100. * correct / total
            }
            
            print(f'検証結果: 平均損失: {metrics["loss"]:.4f}, '
                  f'精度: {metrics["accuracy"]:.2f}%')
            
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