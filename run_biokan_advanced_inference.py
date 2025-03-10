"""
BioKANモデルの高度な推論とOptunaによるハイパーパラメータ最適化
様々な推論タスク（分類、回帰、セグメンテーションなど）に対応した転移学習を行います
"""

import sys
import os

# Pythonバージョンチェック
if sys.version_info < (3, 11):
    print("エラー: Python 3.11以上が必要です")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import argparse
import optuna
from sklearn.datasets import make_classification, make_regression, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
import random

# 自作モジュールのインポート
from biokan_training import EnhancedBioKANModel
from biokan_transfer_learning import (
    TransferBioKANModel, 
    get_dataset, 
    fine_tune_model, 
    evaluate_model,
    visualize_results,
    run_inference
)
from cuda_info_manager import print_cuda_info, get_device, setup_japanese_fonts

# 日本語フォントの設定
setup_japanese_fonts()

# デバイスの設定
device = get_device()

# デバイス情報の初期表示
print_cuda_info()

# ===============================================
# 高度な推論用データセット
# ===============================================

def get_advanced_dataset(task_type, batch_size=32):
    """
    様々な推論タスク用のデータセットを生成または取得
    
    Args:
        task_type: タスクの種類
        batch_size: バッチサイズ
        
    Returns:
        train_loader, val_loader, test_loader のタプル
    """
    if task_type == 'classification':
        # 標準的な分類データセット (Fashion-MNIST)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 訓練データセット
        train_dataset = datasets.FashionMNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # 訓練/検証分割
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # テストデータセット
        test_dataset = datasets.FashionMNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
    elif task_type == 'regression':
        # 回帰データセットの生成
        X, y = make_regression(
            n_samples=2000, 
            n_features=20, 
            n_informative=10, 
            noise=0.1, 
            random_state=42
        )
        
        # データの正規化
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
        
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 訓練/検証/テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # PyTorchのデータセットに変換
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val).unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_test).unsqueeze(1)
        )
        
    elif task_type == 'multivariate_regression':
        # 多変量回帰データセットの生成 (3次元出力)
        X, y1 = make_regression(n_samples=2000, n_features=20, random_state=42)
        _, y2 = make_regression(n_samples=2000, n_features=20, random_state=43)
        _, y3 = make_regression(n_samples=2000, n_features=20, random_state=44)
        
        y = np.column_stack([y1, y2, y3])
        
        # データの正規化
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
        
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)
        
        # 訓練/検証/テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # PyTorchのデータセットに変換
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_test)
        )
        
    elif task_type == 'sequence':
        # 時系列データセットの生成
        # サイン波 + ノイズ
        seq_length = 24
        n_samples = 1000
        
        t = np.linspace(0, 10, seq_length)
        sequences = []
        targets = []
        
        for i in range(n_samples):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            noise_level = np.random.uniform(0.05, 0.2)
            
            # 次の値を予測
            sequence = np.sin(freq * t + phase) + np.random.normal(0, noise_level, seq_length)
            target = np.sin(freq * (t[-1] + 1) + phase)
            
            sequences.append(sequence)
            targets.append(target)
        
        sequences = np.array(sequences).astype(np.float32)
        targets = np.array(targets).astype(np.float32)
        
        # 訓練/検証/テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # PyTorchのデータセットに変換
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val).unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_test).unsqueeze(1)
        )
        
    elif task_type == 'segmentation':
        # MNISTをセグメンテーションタスクとして使用
        # 背景(0)と前景(1)の2クラスセグメンテーション
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 訓練データセット
        mnist_train = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # セグメンテーション用にデータを変換
        # 画像をそのまま入力とし、閾値0.5で2値化したものをマスクとする
        class SegmentationDataset(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                # 閾値0.5で2値化してマスクを作成
                mask = (img > 0).float()
                return img, mask.squeeze(0)  # [1, 28, 28] -> [28, 28]
        
        train_seg_dataset = SegmentationDataset(mnist_train)
        
        # 訓練/検証分割
        train_size = int(0.8 * len(train_seg_dataset))
        val_size = len(train_seg_dataset) - train_size
        train_dataset, val_dataset = random_split(train_seg_dataset, [train_size, val_size])
        
        # テストデータセット
        mnist_test = datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        test_dataset = SegmentationDataset(mnist_test)
        
    elif task_type == 'anomaly_detection':
        # 異常検知データセットの生成
        # 月形のデータを使用し、一方のクラスを正常、もう一方を異常とする
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
        
        # クラス0を正常データ、クラス1を異常データとする
        normal_idx = (y == 0)
        abnormal_idx = (y == 1)
        
        X_normal = X[normal_idx]
        X_abnormal = X[abnormal_idx]
        
        # 訓練データには正常データのみを使用
        X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
        
        # テストデータには正常と異常の両方を含める
        X_test = np.vstack([X[normal_idx][:50], X[abnormal_idx][:50]])
        y_test = np.hstack([np.zeros(50), np.ones(50)])  # 0:正常, 1:異常
        
        # PyTorchのデータセットに変換
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.zeros(len(X_train))  # ダミーラベル
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.zeros(len(X_val))  # ダミーラベル
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_test)
        )
    
    else:
        raise ValueError(f"サポートされていないタスク種類: {task_type}")
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda'
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda'
    )
    
    return train_loader, val_loader, test_loader

# ===============================================
# Optunaによるハイパーパラメータ最適化
# ===============================================

def optimize_hyperparameters(pretrained_model, task_type, train_loader, val_loader, n_trials=30):
    """
    Optunaを使用してハイパーパラメータを最適化する関数
    
    Args:
        pretrained_model: 事前学習済みモデル
        task_type: タスクの種類 ('classification', 'regression', 'multivariate_regression', 'sequence', 'segmentation', 'anomaly_detection')
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        n_trials: 試行回数
        
    Returns:
        最適なハイパーパラメータ辞書
    """
    # デバイスの設定
    device = get_device()
    
    # CUDA情報を静かモードで表示（冗長な出力を避けるため）
    print_cuda_info(verbose=False)
    
    def objective(trial):
        # ハイパーパラメータの候補を定義
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        freeze_layers = trial.suggest_categorical('freeze_layers', [True, False])
        
        # タスク固有のパラメータ
        additional_params = {}
        
        if task_type == 'multivariate_regression':
            output_dim = trial.suggest_int('output_dim', 2, 5)
            additional_params['output_dim'] = output_dim
        
        if task_type == 'segmentation':
            # Segmentationの場合は特別なパラメータ
            additional_params['img_size'] = 28  # MNISTのサイズ
        
        # モデルの設定
        num_classes = 10 if task_type == 'classification' else 2 if task_type == 'segmentation' else 1
        
        # 転移学習モデルの作成
        transfer_model = TransferBioKANModel(
            pretrained_model=pretrained_model,
            task_type=task_type,
            num_classes=num_classes,
            output_dim=1 if task_type == 'regression' else 3 if task_type == 'multivariate_regression' else 1,
            freeze_layers=freeze_layers,
            additional_params=additional_params
        )
        
        # モデルのDropout値を更新
        for module in transfer_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        
        transfer_model = transfer_model.to(device)
        
        # モデルの訓練
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=lr)
        
        # タスクに応じた損失関数
        if task_type in ['classification', 'segmentation']:
            criterion = nn.CrossEntropyLoss()
        elif task_type in ['regression', 'multivariate_regression', 'sequence']:
            criterion = nn.MSELoss()
        elif task_type == 'anomaly_detection':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"サポートされていないタスク種類: {task_type}")
        
        # 早期終了のためのパラメータ
        patience = 5
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # エポック数は少なめに設定
        n_epochs = 10
        
        for epoch in range(n_epochs):
            # 訓練フェーズ
            transfer_model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # タスク固有の入力整形
                if task_type in ['classification', 'regression', 'multivariate_regression', 'anomaly_detection']:
                    if inputs.dim() > 2:
                        inputs = inputs.view(inputs.size(0), -1)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=device.type == 'cuda'):
                    outputs = transfer_model(inputs)
                    
                    # タスク固有の損失計算
                    if task_type == 'segmentation':
                        loss = criterion(outputs, targets.long())
                    elif task_type == 'anomaly_detection':
                        # 自己符号化器の場合、入力と再構成誤差の最小化
                        reconstruction_error = outputs  # 既に誤差として計算済み
                        loss = reconstruction_error.mean()
                    else:
                        loss = criterion(outputs, targets)
                
                # デバイス対応の逆伝播
                loss.backward()
                optimizer.step()
            
            # 検証フェーズ
            transfer_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # タスク固有の入力整形
                    if task_type in ['classification', 'regression', 'multivariate_regression', 'anomaly_detection']:
                        if inputs.dim() > 2:
                            inputs = inputs.view(inputs.size(0), -1)
                    
                    with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=device.type == 'cuda'):
                        outputs = transfer_model(inputs)
                        
                        # タスク固有の損失計算
                        if task_type == 'segmentation':
                            batch_loss = criterion(outputs, targets.long())
                        elif task_type == 'anomaly_detection':
                            batch_loss = outputs.mean()
                        else:
                            batch_loss = criterion(outputs, targets)
                    
                    val_loss += batch_loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # 早期終了の判定
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Optunaへの中間報告
            trial.report(val_loss, epoch)
            
            # プルーニング
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_val_loss
    
    # Optunaの設定と実行
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # 最適なハイパーパラメータ
    best_params = study.best_params
    
    # 最適化結果のプロット
    try:
        optimization_history_plot = optuna.visualization.plot_optimization_history(study)
        param_importance_plot = optuna.visualization.plot_param_importances(study)
        
        # 可視化（Plotlyを使用）
        optimization_history_plot.write_image("optimization_history.png")
        param_importance_plot.write_image("param_importance.png")
        print("最適化結果のプロットを保存しました")
    except Exception as e:
        print(f"プロットの保存に失敗しました: {e}")
    
    # 最良のパラメータでモデルを再構築
    additional_params = {}
    if task_type == 'multivariate_regression' and 'output_dim' in best_params:
        additional_params['output_dim'] = best_params['output_dim']
    if task_type == 'segmentation':
        additional_params['img_size'] = 28
    
    num_classes = 10 if task_type == 'classification' else 2 if task_type == 'segmentation' else 1
    
    best_model = TransferBioKANModel(
        pretrained_model=pretrained_model,
        task_type=task_type,
        num_classes=num_classes,
        output_dim=1 if task_type == 'regression' else 3 if task_type == 'multivariate_regression' else 1,
        freeze_layers=best_params['freeze_layers'],
        additional_params=additional_params
    )
    
    # Dropoutの更新
    for module in best_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = best_params['dropout']
    
    best_model = best_model.to(device)
    
    return best_params, best_model, study

# ===============================================
# メイン実行関数
# ===============================================

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='BioKANモデルの高度な推論とハイパーパラメータ最適化')
    
    # 基本パラメータ
    parser.add_argument('--task-type', type=str, default='classification',
                       choices=['classification', 'regression', 'multivariate_regression', 
                               'sequence', 'segmentation', 'anomaly_detection'],
                       help='推論タスクの種類')
    parser.add_argument('--pretrained-model', type=str, default='biokan_trained_models/best_biokan_model.pth',
                       help='事前学習済みモデルのパス')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=15,
                       help='訓練エポック数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学習率')
    
    # Optuna関連パラメータ
    parser.add_argument('--optimize', action='store_true',
                       help='Optunaでハイパーパラメータ最適化を行う')
    parser.add_argument('--n-trials', type=int, default=30,
                       help='Optunaの試行回数')
    
    # 出力関連パラメータ
    parser.add_argument('--save-dir', type=str, default='transfer_models',
                       help='モデル保存ディレクトリ')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細な出力を表示')
    parser.add_argument('--quiet', action='store_true',
                       help='静かモード（進捗表示の少ない出力）')
    
    args = parser.parse_args()
    
    # CUDA情報の表示（静かモードの設定に合わせる）
    print_cuda_info(verbose=not args.quiet)
    
    # 出力ディレクトリの作成
    os.makedirs(args.save_dir, exist_ok=True)
    
    # タスク情報の表示
    print(f"\n{args.task_type}タスク用のデータセットを準備中...")
    
    # データセットの設定
    train_loader, val_loader, test_loader = get_advanced_dataset(args.task_type, args.batch_size)
    
    # 事前学習済みモデルの読み込み
    pretrained_model = EnhancedBioKANModel().to(device)
    
    # 事前学習済み重みの読み込み
    model_weights = torch.load(args.pretrained_model, map_location=device)
    pretrained_model.load_state_dict(model_weights)
    
    # Optunaによるハイパーパラメータ最適化（引数で指定）
    if args.optimize:
        print(f"\nOptunaによるハイパーパラメータ最適化を開始します（{args.n_trials}回試行）...")
        best_params, best_model, study = optimize_hyperparameters(
            pretrained_model=pretrained_model,
            task_type=args.task_type,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=args.n_trials
        )
        
        # 最適化結果を保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.save_dir, f"best_params_{args.task_type}_{timestamp}.json")
        with open(save_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print(f"\n最適なハイパーパラメータを保存しました: {save_path}")
        
        # 最適なパラメータで転移学習モデルをトレーニング
        print("\n最適なハイパーパラメータでモデルを訓練中...")
        history, optimized_model = fine_tune_model(
            model=best_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=best_params['learning_rate'],
            task_type=args.task_type
        )
        
        # 最適化されたモデルの保存
        model_save_path = os.path.join(args.save_dir, f"optimized_{args.task_type}_model.pth")
        torch.save(optimized_model.state_dict(), model_save_path)
        print(f"最適化されたモデルを保存しました: {model_save_path}")
        
        # ハイパーパラメータの保存
        params_save_path = os.path.join(args.save_dir, f"optimized_{args.task_type}_params.json")
        with open(params_save_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # 評価用のモデル
        eval_model = optimized_model
        
    else:
        print("\nデフォルト設定でモデルを訓練中...")
        # デフォルト設定での転移学習モデル作成
        additional_params = {}
        if args.task_type == 'segmentation':
            additional_params['img_size'] = 28
            
        num_classes = 10 if args.task_type == 'classification' else 2 if args.task_type == 'segmentation' else 1
        
        transfer_model = TransferBioKANModel(
            pretrained_model=pretrained_model,
            task_type=args.task_type,
            num_classes=num_classes,
            output_dim=1 if args.task_type == 'regression' else 3 if args.task_type == 'multivariate_regression' else 1,
            freeze_layers=True,
            additional_params=additional_params
        )
        
        transfer_model = transfer_model.to(device)
        
        # モデルの訓練
        history, trained_model = fine_tune_model(
            model=transfer_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            task_type=args.task_type
        )
        
        # モデルの保存
        model_save_path = os.path.join(args.save_dir, f"transfer_{args.task_type}_model.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"訓練済みモデルを保存しました: {model_save_path}")
        
        # 評価用のモデル
        eval_model = trained_model
    
    # モデルの評価
    print("\nテストデータでモデルを評価中...")
    metrics = evaluate_model(eval_model, test_loader, task_type=args.task_type)
    
    # 結果の表示
    print("\n評価結果:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, np.ndarray):
            continue  # 混同行列などの大きなデータは表示しない
        print(f"  {metric_name}: {metric_value}")
    
    # 結果の保存
    results_save_path = os.path.join(args.save_dir, f"{args.task_type}_results.json")
    with open(results_save_path, 'w') as f:
        # NumPy配列をリストに変換
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = v
        
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"評価結果を保存しました: {results_save_path}")
    
    # サンプル推論の実行
    print("\nサンプル推論を実行中...")
    for i, (data, target) in enumerate(test_loader):
        if i >= 1:  # 1サンプルのみ
            break
            
        data = data.to(device)
        
        explanation = eval_model.explain_prediction(data[0:1])
        
        print("\n推論結果:")
        for key, value in explanation.items():
            if isinstance(value, np.ndarray):
                if value.size < 10:  # 小さい配列のみ表示
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: [shape: {value.shape}]")
            else:
                print(f"  {key}: {value}")
    
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main() 