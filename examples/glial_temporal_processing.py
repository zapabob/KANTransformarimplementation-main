#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
グリア細胞を活用した皮質層間時間差処理のサンプル実装
時系列データに対するBioKANモデルの適用例
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# プロジェクトルートへのパス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biokan.core.biokan_model import create_biokan_classifier
from biokan.utils.data_generator import generate_temporal_data
from biokan.utils.visualization import visualize_layer_activations, plot_astrocyte_calcium


def train_temporal_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """
    時系列データに対してBioKANモデルを訓練する
    
    Args:
        model: BioKANモデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        epochs: エポック数
        lr: 学習率
        device: 計算デバイス
    
    Returns:
        訓練履歴
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 訓練履歴
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 勾配リセット
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 逆伝播
            loss.backward()
            optimizer.step()
            
            # 統計
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # エポック平均の訓練損失
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # エポック平均の検証損失
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 履歴に追加
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'エポック {epoch+1}/{epochs} - '
              f'損失: {train_loss:.4f} - 精度: {train_acc:.2f}% - '
              f'検証損失: {val_loss:.4f} - 検証精度: {val_acc:.2f}%')
    
    return history


def visualize_astrocyte_activity(model, sample_data):
    """
    アストロサイト活動の可視化
    
    Args:
        model: 訓練済みBioKANモデル
        sample_data: サンプルデータ [batch_size, in_features]
    """
    # 推論モード
    model.eval()
    
    # 順伝播を実行
    with torch.no_grad():
        _ = model(sample_data)
    
    # アストロサイトの活動を可視化
    if hasattr(model, 'astrocytes'):
        # 各層のアストロサイトのカルシウム活動をプロット
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, astrocyte in enumerate(model.astrocytes):
            ax = axes[i]
            
            # カルシウム活性化状態を取得
            calcium_activity = astrocyte.activation
            
            # ヒートマップとしてプロット
            im = ax.imshow(calcium_activity, cmap='hot', interpolation='nearest')
            ax.set_title(f'層 {i+1} アストロサイト Ca²⁺ 活動')
            fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('astrocyte_calcium_activity.png')
        plt.close()
        
        # 皮質層間時間差変調効果を可視化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, astrocyte in enumerate(model.astrocytes):
            ax = axes[i]
            
            # 時間差変調効果を取得
            temporal_mod = astrocyte.get_modulatory_effect()['cross_layer_temporal_modulation']
            
            # ヒートマップとしてプロット
            im = ax.imshow(temporal_mod, cmap='coolwarm', interpolation='nearest')
            ax.set_title(f'層 {i+1} 時間差変調効果')
            fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('temporal_modulation_effect.png')
        plt.close()


def main():
    """メイン実行関数"""
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用デバイス: {device}')
    
    # 時系列データ生成
    # 2クラス分類問題の時系列データを生成
    # 各シーケンスはt0とt1の時間差でパターンが現れる場合にクラス1、そうでなければクラス0
    print('時系列データの生成中...')
    X, y = generate_temporal_data(
        n_samples=1000,
        sequence_length=20,
        n_features=64,
        n_classes=2,
        temporal_dependency=True,
        random_state=42
    )
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # PyTorchテンソルに変換
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # データローダー作成
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # モデル作成
    # グリア細胞の皮質層間時間差処理を有効化
    in_features = X.shape[1]  # 特徴数
    
    print('BioKANモデル（グリア細胞時間差処理あり）の作成...')
    biokan_model = create_biokan_classifier(
        in_features=in_features,
        hidden_dim=128,
        num_classes=2,
        num_blocks=3,
        attention_type='biological',  # 生物学的アテンション
        neuromodulation=True  # グリア細胞と神経調節を有効化
    )
    
    # 比較用の標準モデル（神経調節なし）
    print('比較用モデル（グリア細胞なし）の作成...')
    standard_model = create_biokan_classifier(
        in_features=in_features,
        hidden_dim=128,
        num_classes=2,
        num_blocks=3,
        attention_type='hierarchical',  # 階層的アテンション
        neuromodulation=False  # グリア細胞と神経調節を無効化
    )
    
    # グリア細胞モデルの訓練
    print('\nグリア細胞モデルの訓練開始...')
    biokan_history = train_temporal_model(
        biokan_model,
        train_loader,
        test_loader,
        epochs=10,
        lr=0.001,
        device=device
    )
    
    # 標準モデルの訓練
    print('\n標準モデルの訓練開始...')
    standard_history = train_temporal_model(
        standard_model,
        train_loader,
        test_loader,
        epochs=10,
        lr=0.001,
        device=device
    )
    
    # 結果の可視化
    print('\n訓練結果の可視化...')
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(biokan_history['train_acc'], label='グリア細胞モデル 訓練')
    plt.plot(biokan_history['val_acc'], label='グリア細胞モデル 検証')
    plt.plot(standard_history['train_acc'], label='標準モデル 訓練', linestyle='--')
    plt.plot(standard_history['val_acc'], label='標準モデル 検証', linestyle='--')
    plt.title('モデル精度比較')
    plt.xlabel('エポック')
    plt.ylabel('精度 (%)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(biokan_history['train_loss'], label='グリア細胞モデル 訓練')
    plt.plot(biokan_history['val_loss'], label='グリア細胞モデル 検証')
    plt.plot(standard_history['train_loss'], label='標準モデル 訓練', linestyle='--')
    plt.plot(standard_history['val_loss'], label='標準モデル 検証', linestyle='--')
    plt.title('モデル損失比較')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('temporal_model_comparison.png')
    plt.close()
    
    # アストロサイト活動の可視化
    print('アストロサイト活動の可視化...')
    sample_batch = next(iter(test_loader))[0].to(device)
    visualize_astrocyte_activity(biokan_model, sample_batch)
    
    print('実行完了。結果は画像ファイルとして保存されました。')


if __name__ == '__main__':
    main() 