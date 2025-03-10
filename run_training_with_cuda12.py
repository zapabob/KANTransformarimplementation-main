"""
CUDA 12に最適化されたBioKANモデルの訓練スクリプト
tqdmによる進捗表示と決定係数の可視化機能を含む
"""

import sys

# Pythonバージョンチェック
if sys.version_info < (3, 11):
    print("エラー: Python 3.11以上が必要です")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import argparse

# BioKAN訓練モジュールのインポート
from biokan_training import EnhancedBioKANModel, train_enhanced_biokan, calculate_r_squared

def print_system_info():
    """システム情報の表示"""
    print("=" * 50)
    print("システム情報:")
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"Python バージョン: {sys.version}")
    
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"CUDA バージョン: {cuda_version}")
        
        # CUDA 12の互換性チェック
        if cuda_version.startswith('12.'):
            print("✓ CUDA 12が検出されました。最適化されたトレーニングを行います。")
        else:
            print(f"! 警告: CUDA {cuda_version}が検出されました。このスクリプトはCUDA 12向けに最適化されています。")
        
        # GPU情報
        device_count = torch.cuda.device_count()
        print(f"利用可能なGPUデバイス数: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(f"GPU {i}: {device_name} (Compute Capability: {device_capability[0]}.{device_capability[1]})")
            
            # メモリ情報
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  合計メモリ: {total_memory:.2f} GB")
    else:
        print("! 警告: GPUが検出されませんでした。CPUで実行します。")
    
    print("=" * 50)

def load_mnist_dataset(batch_size=64):
    """MNISTデータセットの読み込み"""
    # データ変換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 訓練データのダウンロードと変換
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 訓練データと検証データに分割
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # テストデータのダウンロードと変換
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

def parse_arguments():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description='CUDA 12対応のBioKANモデル訓練スクリプト')
    
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='バッチサイズ (デフォルト: 64)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='訓練エポック数 (デフォルト: 10)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='学習率 (デフォルト: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5, 
                        help='重み減衰 (デフォルト: 1e-5)')
    parser.add_argument('--hidden-dim', type=int, default=128, 
                        help='隠れ層の次元 (デフォルト: 128)')
    parser.add_argument('--num-blocks', type=int, default=3, 
                        help='BioKANブロック数 (デフォルト: 3)')
    parser.add_argument('--save-dir', type=str, default='biokan_trained_models', 
                        help='モデル保存ディレクトリ (デフォルト: biokan_trained_models)')
    parser.add_argument('--gpu-clear-cache', action='store_true', 
                        help='各エポック後にGPUキャッシュをクリア')
    
    return parser.parse_args()

def main():
    """メイン実行関数"""
    # コマンドライン引数のパース
    args = parse_arguments()
    
    # システム情報の表示
    print_system_info()
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセットの読み込み
    print("\nMNISTデータセットを読み込んでいます...")
    train_loader, val_loader, test_loader = load_mnist_dataset(batch_size=args.batch_size)
    print(f"データセット読み込み完了")
    
    # モデルの作成
    print("\nBioKANモデルを作成しています...")
    model = EnhancedBioKANModel(
        in_features=784,  # MNIST: 28x28=784
        hidden_dim=args.hidden_dim,
        num_classes=10,   # MNIST: 0-9の10クラス
        num_blocks=args.num_blocks
    ).to(device)
    
    print(f"モデル作成完了: 隠れ層次元={args.hidden_dim}, ブロック数={args.num_blocks}")
    
    # モデルのパラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"合計パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")
    
    # 訓練設定の表示
    print("\n訓練設定:")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"エポック数: {args.epochs}")
    print(f"学習率: {args.lr}")
    print(f"重み減衰: {args.weight_decay}")
    print(f"モデル保存先: {args.save_dir}")
    print(f"GPUキャッシュクリア: {'有効' if args.gpu_clear_cache else '無効'}")
    
    # モデルの訓練
    print("\n訓練を開始します...")
    history, best_model_path = train_enhanced_biokan(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        gpu_cache_clear=args.gpu_clear_cache
    )
    
    # 最良モデルのロード
    print(f"\n最良モデルをロードしています: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    
    # テストデータでの評価
    print("\nテストデータでの評価:")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_outputs_all = []
    test_targets_all = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten
            
            # 前向き計算
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 決定係数計算用
            test_outputs_all.append(outputs)
            test_targets_all.append(targets)
            
            # 統計更新
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    # テスト指標の計算
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total
    
    # 決定係数の計算
    test_outputs_all = torch.cat(test_outputs_all, dim=0)
    test_targets_all = torch.cat(test_targets_all, dim=0)
    test_r2 = calculate_r_squared(test_targets_all, test_outputs_all)
    
    # 結果の表示
    print(f"テスト損失: {test_loss:.4f}")
    print(f"テスト精度: {test_acc:.4f}")
    print(f"テスト決定係数(R²): {test_r2:.4f}")
    
    # 結果のJSONファイルへの保存
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_r_squared": test_r2,
        "model_params": {
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        "training_params": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay
        },
        "final_metrics": {
            "train_loss": history["train_loss"][-1],
            "train_acc": history["train_acc"][-1],
            "train_r2": history["train_r2"][-1],
            "val_loss": history["val_loss"][-1],
            "val_acc": history["val_acc"][-1],
            "val_r2": history["val_r2"][-1]
        }
    }
    
    # 結果の保存
    results_file = os.path.join(args.save_dir, "test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n結果を保存しました: {results_file}")
    print("\n訓練が完了しました。✓")

if __name__ == "__main__":
    main() 