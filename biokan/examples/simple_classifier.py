"""
BioKANモデルを使用した簡単な分類器の例
MNISTデータセットを使用して画像分類を行う
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from typing import Dict, List, Tuple, Optional, Any

from biokan.core.biokan_model import create_biokan_classifier
from biokan.xai.explainers import BioKANExplainer
from biokan.visualization.visualizers import visualize_explanation


def train_model(model: nn.Module, 
               train_loader: DataLoader, 
               val_loader: DataLoader, 
               device: torch.device,
               epochs: int = 10,
               learning_rate: float = 0.001,
               weight_decay: float = 1e-5,
               save_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    モデルを訓練する
    
    Args:
        model: 訓練するモデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        device: 計算に使用するデバイス
        epochs: エポック数
        learning_rate: 学習率
        weight_decay: 重み減衰
        save_path: モデルの保存パス（オプション）
        
    Returns:
        訓練履歴
    """
    # 損失関数とオプティマイザーの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学習率スケジューラー
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # 訓練履歴
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 最良のモデルを追跡
    best_val_loss = float('inf')
    
    print("訓練開始...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 勾配をゼロに
            optimizer.zero_grad()
            
            # フォワードパス
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # バックワードパス
            loss.backward()
            optimizer.step()
            
            # 統計を更新
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 進捗表示
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"\rエポック [{epoch+1}/{epochs}] バッチ [{batch_idx+1}/{len(train_loader)}] "
                     f"損失: {loss.item():.4f} 精度: {100.*train_correct/train_total:.2f}%", end="")
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # フォワードパス
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 統計を更新
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 学習率を調整
        scheduler.step(val_loss)
        
        # 履歴を記録
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 現在のエポックの結果を表示
        elapsed_time = time.time() - start_time
        print(f"\rエポック [{epoch+1}/{epochs}] "
             f"訓練損失: {train_loss:.4f} 訓練精度: {100.*train_acc:.2f}% "
             f"検証損失: {val_loss:.4f} 検証精度: {100.*val_acc:.2f}% "
             f"時間: {elapsed_time:.2f}秒")
        
        # 最良のモデルを保存
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"モデル保存: {save_path}")
    
    return history


def explain_predictions(model: nn.Module, 
                      test_loader: DataLoader, 
                      device: torch.device,
                      num_samples: int = 5,
                      save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    モデルの予測を説明する
    
    Args:
        model: 説明するモデル
        test_loader: テストデータローダー
        device: 計算に使用するデバイス
        num_samples: 説明するサンプル数
        save_path: 図の保存パス（オプション）
        
    Returns:
        説明情報
    """
    # 評価モードに設定
    model.eval()
    
    # 説明者を初期化
    explainer = BioKANExplainer(model)
    
    # いくつかのサンプルを取得
    samples = []
    for inputs, targets in test_loader:
        batch_samples = [(inputs[i], targets[i]) for i in range(len(targets))]
        samples.extend(batch_samples)
        if len(samples) >= num_samples:
            break
    
    samples = samples[:num_samples]
    
    # 各サンプルを説明
    explanations = []
    for i, (input_data, target) in enumerate(samples):
        # 入力データを前処理
        input_tensor = input_data.unsqueeze(0).to(device)  # バッチ次元を追加
        target_tensor = target.unsqueeze(0).to(device)     # バッチ次元を追加
        
        # 説明を生成
        explanation = explainer.analyze(input_tensor, target_tensor)
        
        # 予測を取得
        with torch.no_grad():
            output = model(input_tensor)
            
        _, predicted = output.max(1)
        
        # 説明情報を拡張
        explanation['input_data'] = input_data.cpu().numpy()
        explanation['target'] = target.item()
        explanation['predicted'] = predicted.item()
        
        # 反事実的説明（異なるクラスに変更する）
        if predicted.item() == target.item():
            # 正解の場合、異なるクラスを目標に設定
            other_class = (target.item() + 1) % 10  # MNISTは10クラス
            
            # 反事実的説明を生成
            counterfactual = explainer.create_counterfactual(
                input_tensor, 
                target_class=other_class,
                max_iterations=50
            )
            
            explanation['counterfactual'] = counterfactual
        
        explanations.append(explanation)
        
        # 説明を可視化
        figures = visualize_explanation(explanation)
        
        # 図を保存（オプション）
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            for fig_name, fig in figures.items():
                fig.savefig(f"{save_path}/sample_{i}_{fig_name}.png")
                plt.close(fig)
    
    return {'explanations': explanations}


def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='BioKANを使用したMNIST分類器の訓練と説明')
    parser.add_argument('--batch_size', type=int, default=128, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=5, help='エポック数')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隠れ層の次元')
    parser.add_argument('--num_blocks', type=int, default=2, help='BioKANブロック数')
    parser.add_argument('--attention_type', type=str, default='biological', 
                      choices=['biological', 'cortical', 'hierarchical'], help='アテンションタイプ')
    parser.add_argument('--neuromodulation', action='store_true', default=True, help='神経調節を有効にする')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='出力ディレクトリ')
    parser.add_argument('--explain', action='store_true', help='予測の説明を生成する')
    parser.add_argument('--num_explain', type=int, default=5, help='説明するサンプル数')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.save_dir, exist_ok=True)
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセットの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # データローダーの作成
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # モデルの作成
    model = create_biokan_classifier(
        in_features=28*28,  # MNIST画像サイズ
        hidden_dim=args.hidden_dim,
        num_classes=10,     # MNISTは10クラス
        num_blocks=args.num_blocks,
        attention_type=args.attention_type,
        neuromodulation=args.neuromodulation
    )
    
    # モデルをデバイスに移動
    model = model.to(device)
    
    # モデルの情報を表示
    print(f"モデル構成:")
    print(f"  隠れ層次元: {args.hidden_dim}")
    print(f"  BioKANブロック数: {args.num_blocks}")
    print(f"  アテンションタイプ: {args.attention_type}")
    print(f"  神経調節: {'有効' if args.neuromodulation else '無効'}")
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters())}")
    
    # モデルの訓練
    model_save_path = f"{args.save_dir}/biokan_mnist.pth"
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        device,
        epochs=args.epochs,
        learning_rate=args.lr,
        save_path=model_save_path
    )
    
    # 学習曲線のプロット
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='訓練損失')
    plt.plot(history['val_loss'], label='検証損失')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.legend()
    plt.title('学習曲線 - 損失')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='訓練精度')
    plt.plot(history['val_acc'], label='検証精度')
    plt.xlabel('エポック')
    plt.ylabel('精度')
    plt.legend()
    plt.title('学習曲線 - 精度')
    
    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/learning_curves.png")
    
    # モデルの評価
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 入力データの形状変換（MNISTの場合）
            batch_size = inputs.size(0)
            inputs = inputs.view(batch_size, -1)  # [batch_size, 784]
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    print(f"テスト損失: {test_loss:.4f} テスト精度: {100.*test_acc:.2f}%")
    
    # 説明の生成（オプション）
    if args.explain:
        explanation_save_path = f"{args.save_dir}/explanations"
        explanations = explain_predictions(
            model, 
            test_loader, 
            device,
            num_samples=args.num_explain,
            save_path=explanation_save_path
        )
        
        print(f"説明が生成されました: {explanation_save_path}")


if __name__ == "__main__":
    main() 