"""
生物学的な機構を組み込んだMNIST学習スクリプト
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from typing import Dict, Optional, Tuple

from biokan.neuro.neurotransmitter_dynamics import DetailedNeurotransmitterSystem
from biokan.neuro.synaptic_plasticity import SynapticPlasticityModule
from biokan.neuro.neural_oscillations import NeuralOscillator, BrainwaveAnalyzer
from biokan.learning.biological_learning import BiologicalOptimizer, RewardModulatedSTDP

class EnhancedBioKANClassifier(nn.Module):
    """
    生物学的な機構を組み込んだBioKANモデル
    """
    
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 入力層
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 神経伝達物質システム
        self.neurotransmitter_system = DetailedNeurotransmitterSystem(hidden_dim)
        
        # シナプス可塑性
        self.plasticity = SynapticPlasticityModule(hidden_dim)
        
        # 神経振動子
        self.oscillator = NeuralOscillator(hidden_dim)
        
        # 報酬調節STDP
        self.reward_stdp = RewardModulatedSTDP(hidden_dim)
        
        # 出力層
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
        # 脳波解析
        self.brainwave_analyzer = BrainwaveAnalyzer()
        
        # 活動履歴
        self.activity_history = []
        
        # ドロップアウト
        self.dropout = nn.Dropout(0.1)
        
        # 学習率の調整
        self.learning_rate = 0.0001  # より小さな学習率
        
    def forward(self, x: torch.Tensor, reward: float = 0.0) -> Tuple[torch.Tensor, Dict]:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            reward: 報酬信号
            
        Returns:
            output: 分類結果
            info: 神経活動情報
        """
        # 入力層
        h = self.input_layer(x)
        
        # ドロップアウト
        h = self.dropout(h)
        
        # 神経伝達物質の調節を適用
        modulation = self.neurotransmitter_system(h)
        
        # シナプス可塑性を適用
        h = self.plasticity(h, modulation)
        
        # 神経振動を重畳
        h = self.oscillator(h, modulation=modulation)
        
        # 報酬調節STDPを適用
        h = self.reward_stdp(h, h, reward=reward, modulation=modulation)
        
        # 出力層
        output = self.output_layer(h)
        
        # 活動履歴を更新
        self.activity_history.append(h.detach().mean().item())
        if len(self.activity_history) > 1000:
            self.activity_history.pop(0)
        
        # 脳波解析
        if len(self.activity_history) >= 100:
            signal = torch.tensor(self.activity_history[-100:])
            brain_waves = self.brainwave_analyzer.analyze_power_spectrum(signal)
        else:
            brain_waves = {}
            
        # 神経活動情報を収集
        info = {
            'modulation': modulation,
            'brain_waves': brain_waves,
            'mean_activity': h.mean().item(),
            'sparsity': (h > 0).float().mean().item()
        }
        
        return output, info

    def monitor_model_state(self):
        metrics = {
            'weight_norm': torch.norm(self.output_layer.weight).item(),
            'meta_plasticity_mean': self.plasticity.meta_plasticity.mean().item(),
            'activity_level': self.activity_history.mean().item(),
            'gradient_magnitude': torch.norm(self.output_layer.weight.grad).item() if self.output_layer.weight.grad is not None else 0
        }
        return metrics

    def update_meta_plasticity(self, activity: torch.Tensor) -> None:
        # メタ可塑性の制御強化
        self.meta_learning_rate = nn.Parameter(torch.tensor(0.0005))  # より慎重な更新
        self.plasticity.meta_plasticity.data = torch.clamp(self.plasticity.meta_plasticity, 0.1, 1.5)  # より狭い範囲

    def update_homeostatic(self, activity: torch.Tensor) -> None:
        # より強力なホメオスタティック制御
        self.target_activity = nn.Parameter(torch.tensor(0.05))  # より低い目標活性化
        self.homeostatic_rate = nn.Parameter(torch.tensor(0.02))  # より強い制御

    def compute_regularization(self):
        # L2正則化
        return 0.01 * torch.norm(self.output_layer.weight)

    def update_stdp(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        # ...
        update = torch.clamp(update, -1.0, 1.0)  # 更新量の制限
        self.output_layer.weight.data += update

def train_enhanced_mnist(
    batch_size: int = 64,
    epochs: int = 10,
    hidden_dim: int = 128,
    learning_rate: float = 0.001,
    device: str = 'cuda'
):
    """
    拡張されたMNIST学習を実行
    
    Args:
        batch_size: バッチサイズ
        epochs: エポック数
        hidden_dim: 隠れ層の次元
        learning_rate: 学習率
        device: 使用デバイス
    """
    # データセットの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # モデルの初期化
    model = EnhancedBioKANClassifier(
        in_features=28*28,
        hidden_dim=hidden_dim,
        num_classes=10
    ).to(device)
    
    # 生物学的な最適化アルゴリズム
    optimizer = BiologicalOptimizer(
        model.parameters(),
        lr=learning_rate,
        hebbian_rate=0.001,
        homeostatic_rate=0.0001
    )
    
    # 損失関数
    criterion = nn.CrossEntropyLoss()
    
    # 学習履歴
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'brain_waves': [],
        'neuromodulation': []
    }
    
    # 学習ループ
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        
        # エポックごとの学習
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # 順伝播
            reward = 0.0  # 初期報酬
            outputs, info = model(images, reward)
            loss = criterion(outputs, labels)
            
            # 報酬の計算（正解率に基づく）
            _, predicted = torch.max(outputs.data, 1)
            batch_accuracy = (predicted == labels).float().mean().item()
            reward = batch_accuracy - 0.5  # 報酬のベースラインを0.5とする
            
            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(modulation=info['modulation'])
            
            # 統計の更新
            train_loss += loss.item()
            train_correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # 学習履歴に神経活動情報を追加
            if batch_idx % 100 == 0:
                history['brain_waves'].append(info['brain_waves'])
                history['neuromodulation'].append(info['modulation'])
        
        # エポックの統計
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / total
        
        # テストデータでの評価
        test_loss, test_acc = evaluate_enhanced_model(model, test_loader, criterion, device)
        
        # 履歴の更新
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # モデルの保存
        if epoch == 0 or test_acc > max(history['test_acc'][:-1]):
            torch.save(model.state_dict(), 'best_enhanced_biokan_mnist.pth')
            
    return model, history

def evaluate_enhanced_model(model, test_loader, criterion, device):
    """
    拡張モデルの評価
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = correct / total
    
    return test_loss, test_acc

def plot_enhanced_training_history(history):
    """
    拡張された学習履歴のプロット
    """
    # 精度と損失のプロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 精度のプロット
    ax1.plot(history['train_acc'], label='Train Accuracy')
    ax1.plot(history['test_acc'], label='Test Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 損失のプロット
    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['test_loss'], label='Test Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_mnist_training_history.png')
    plt.close()
    
    # 脳波パワーの時間変化をプロット
    if history['brain_waves']:
        fig, ax = plt.subplots(figsize=(12, 6))
        waves = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        for wave in waves:
            powers = [bw.get(wave, 0) for bw in history['brain_waves']]
            ax.plot(powers, label=wave)
        
        ax.set_title('Brain Wave Power Over Training')
        ax.set_xlabel('Training Steps (x100)')
        ax.set_ylabel('Power')
        ax.legend()
        plt.savefig('brain_wave_dynamics.png')
        plt.close()

if __name__ == '__main__':
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 学習の実行
    model, history = train_enhanced_mnist(
        batch_size=64,
        epochs=10,
        hidden_dim=128,
        learning_rate=0.001,
        device=device
    )
    
    # 結果のプロット
    plot_enhanced_training_history(history) 