"""
CIFAR-10データセットでOptunaを使ったBioKANモデルのハイパーパラメータ最適化
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
import json
import optuna
from optuna.trial import Trial
from typing import Dict, List, Tuple, Optional, Any

from biokan.core.biokan_model import create_biokan_classifier


def objective(trial: Trial, device: torch.device, data_loaders: Dict[str, DataLoader], 
             n_epochs: int = 20, save_dir: str = "./optuna_results") -> float:
    """
    Optunaの目的関数 - ハイパーパラメータを受け取り、検証損失を返す
    
    Args:
        trial: Optunaのトライアル
        device: 計算に使用するデバイス
        data_loaders: データローダー（'train'と'val'のキーを持つ辞書）
        n_epochs: エポック数
        save_dir: 結果保存ディレクトリ
        
    Returns:
        検証損失（最小化する値）
    """
    # ハイパーパラメータのサンプリング
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
    num_blocks = trial.suggest_int("num_blocks", 2, 5)
    attention_type = trial.suggest_categorical("attention_type", ["biological", "cortical", "hierarchical"])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    neuromodulation = trial.suggest_categorical("neuromodulation", [True, False])
    
    # モデルの作成
    model = create_biokan_classifier(
        in_features=3*32*32,  # CIFAR-10画像サイズ
        hidden_dim=hidden_dim,
        num_classes=10,      # CIFAR-10は10クラス
        num_blocks=num_blocks,
        attention_type=attention_type,
        dropout=dropout,
        neuromodulation=neuromodulation
    )
    
    # モデルをデバイスに移動
    model = model.to(device)
    
    # 損失関数とオプティマイザーの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学習率スケジューラー
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # 訓練履歴
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 最良の検証損失を追跡
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # 各エポックでの訓練
    for epoch in range(n_epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(data_loaders['train']):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 入力データの形状変換（CIFAR-10の場合）
            batch_size = inputs.size(0)
            inputs = inputs.view(batch_size, -1)  # [batch_size, 3*32*32]
            
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
        
        train_loss = train_loss / len(data_loaders['train'].dataset)
        train_acc = train_correct / train_total
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loaders['val']:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 入力データの形状変換
                batch_size = inputs.size(0)
                inputs = inputs.view(batch_size, -1)  # [batch_size, 3*32*32]
                
                # フォワードパス
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 統計を更新
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(data_loaders['val'].dataset)
        val_acc = val_correct / val_total
        
        # 学習率を調整
        scheduler.step(val_loss)
        
        # 履歴を記録
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 最良のモデルを更新
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            
            # 最良のモデルと指標を保存
            trial_dir = os.path.join(save_dir, f"trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            
            # モデルの保存
            model_path = os.path.join(trial_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            
            # パラメータと結果の保存
            params = {
                'hidden_dim': hidden_dim,
                'num_blocks': num_blocks,
                'attention_type': attention_type,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'neuromodulation': neuromodulation,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'epoch': epoch
            }
            
            with open(os.path.join(trial_dir, "params.json"), "w") as f:
                json.dump(params, f, indent=4)
        
        # 中間結果をOptunaに報告
        trial.report(val_loss, epoch)
        
        # Optunaによる早期停止
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_loss


def save_optuna_results(study: optuna.Study, save_dir: str):
    """
    Optunaの結果を保存する
    
    Args:
        study: Optunaの最適化結果
        save_dir: 保存先ディレクトリ
    """
    # 最適なパラメータの保存
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    
    results = {
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': len(study.trials),
        'all_params': [t.params for t in study.trials],
        'all_values': [t.value for t in study.trials if t.value is not None],
    }
    
    with open(os.path.join(save_dir, "optuna_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # 重要度の可視化
    if len(study.trials) > 5:  # 十分なトライアル数がある場合
        try:
            importance = optuna.importance.get_param_importances(study)
            importance_data = {k: float(v) for k, v in importance.items()}
            
            with open(os.path.join(save_dir, "param_importance.json"), "w") as f:
                json.dump(importance_data, f, indent=4)
            
            # 重要度のプロット
            plt.figure(figsize=(10, 6))
            plt.barh(list(importance.keys()), list(importance.values()))
            plt.xlabel('重要度')
            plt.title('ハイパーパラメータの重要度')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "param_importance.png"))
            plt.close()
        except Exception as e:
            print(f"パラメータ重要度の計算でエラーが発生しました: {e}")
    
    # 最適化の履歴プロット
    plt.figure(figsize=(10, 6))
    
    # 試行の値をプロット
    trial_values = [t.value for t in study.trials if t.value is not None]
    best_values = np.minimum.accumulate(trial_values)
    
    plt.plot(trial_values, 'o-', color='blue', alpha=0.3, label='試行の値')
    plt.plot(best_values, 'o-', color='red', label='最良値')
    
    plt.axhline(y=best_value, color='green', linestyle='--', label=f'最良値: {best_value:.4f}')
    
    plt.xlabel('試行回数')
    plt.ylabel('検証損失')
    plt.title('Optunaによる最適化の履歴')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimization_history.png"))
    plt.close()
    
    # パラメータの分布プロット
    if len(study.trials) > 5:
        try:
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(os.path.join(save_dir, "parallel_coordinate.png"))
        except Exception as e:
            print(f"並行座標プロットの生成でエラーが発生しました: {e}")
        
        try:
            fig = optuna.visualization.plot_contour(study)
            fig.write_image(os.path.join(save_dir, "contour.png"))
        except Exception as e:
            print(f"等高線プロットの生成でエラーが発生しました: {e}")


def evaluate_best_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """
    最良のモデルをテストデータで評価
    
    Args:
        model: 評価するモデル
        test_loader: テストデータローダー
        device: 計算に使用するデバイス
        
    Returns:
        評価結果（テスト損失と精度）
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 入力データの形状変換
            batch_size = inputs.size(0)
            inputs = inputs.view(batch_size, -1)  # [batch_size, 3*32*32]
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # クラスごとの精度を計算
            for c in range(10):
                class_mask = targets == c
                class_correct[c] += (predicted[class_mask] == c).sum().item()
                class_total[c] += class_mask.sum().item()
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    # クラスごとの精度
    class_acc = {}
    for i in range(10):
        if class_total[i] > 0:
            class_acc[f'class_{i}'] = class_correct[i] / class_total[i]
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'class_accuracy': class_acc
    }


def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='CIFAR-10データセットでOptunaを使ったBioKANモデルのハイパーパラメータ最適化')
    parser.add_argument('--batch_size', type=int, default=128, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=20, help='各トライアルのエポック数')
    parser.add_argument('--n_trials', type=int, default=20, help='Optunaのトライアル数')
    parser.add_argument('--save_dir', type=str, default='./optuna_cifar', help='結果の保存先ディレクトリ')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--eval_best', action='store_true', help='最良のモデルをテストデータで評価する')
    
    args = parser.parse_args()
    
    # 乱数シードの設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 出力ディレクトリの作成
    os.makedirs(args.save_dir, exist_ok=True)
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセットの準備
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)
    
    # データローダーの作成
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 検証データセットに正しい変換を適用
    val_dataset.dataset.transform = transform_test
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # データローダーの辞書
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # Optunaのストレージ設定
    storage_name = f"sqlite:///{os.path.join(args.save_dir, 'optuna.db')}"
    
    # Optuna Studyの作成
    study = optuna.create_study(
        study_name="biokan_cifar10",
        storage=storage_name,
        direction="minimize",  # 検証損失を最小化
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    # 目的関数を部分適用して、Optunaのトライアルのみを受け取るようにする
    objective_func = lambda trial: objective(
        trial, device, data_loaders, args.epochs, args.save_dir
    )
    
    # 最適化の実行
    print(f"Optunaによる最適化を開始します（{args.n_trials}トライアル）...")
    study.optimize(objective_func, n_trials=args.n_trials)
    
    print("最適化が完了しました。")
    print(f"最良のトライアル: {study.best_trial.number}")
    print(f"最良の値（検証損失）: {study.best_value:.4f}")
    print(f"最良のハイパーパラメータ: {study.best_params}")
    
    # 結果の保存
    save_optuna_results(study, args.save_dir)
    
    # 最良のモデルを評価（オプション）
    if args.eval_best:
        print("最良のモデルをテストデータで評価します...")
        
        # 最良のパラメータでモデルを再作成
        best_params = study.best_params
        
        best_model = create_biokan_classifier(
            in_features=3*32*32,
            hidden_dim=best_params['hidden_dim'],
            num_classes=10,
            num_blocks=best_params['num_blocks'],
            attention_type=best_params['attention_type'],
            dropout=best_params['dropout'],
            neuromodulation=best_params['neuromodulation']
        )
        
        # モデルをデバイスに移動
        best_model = best_model.to(device)
        
        # 最良のモデルのパラメータをロード
        best_model_path = os.path.join(args.save_dir, f"trial_{study.best_trial.number}", "best_model.pth")
        
        if os.path.exists(best_model_path):
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            # テストデータで評価
            test_results = evaluate_best_model(best_model, test_loader, device)
            
            print(f"テスト損失: {test_results['test_loss']:.4f}")
            print(f"テスト精度: {test_results['test_accuracy'] * 100:.2f}%")
            
            # クラスごとの精度を表示
            print("クラスごとの精度:")
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
            
            for i, class_name in enumerate(classes):
                class_acc = test_results['class_accuracy'].get(f'class_{i}', 0)
                print(f"  {class_name}: {100.*class_acc:.2f}%")
            
            # クラスごとの精度をプロット
            plt.figure(figsize=(10, 6))
            class_accs = [test_results['class_accuracy'].get(f'class_{i}', 0) for i in range(10)]
            plt.bar(classes, class_accs)
            plt.xlabel('クラス')
            plt.ylabel('精度')
            plt.title('クラスごとの精度')
            plt.xticks(rotation=45)
            plt.ylim(0, 1.0)
            for i, v in enumerate(class_accs):
                plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "class_accuracy.png"))
            plt.close()
            
            # テスト結果の保存
            with open(os.path.join(args.save_dir, "test_results.json"), "w") as f:
                json.dump(test_results, f, indent=4)
        else:
            print(f"警告: 最良のモデル {best_model_path} が見つかりません。")


if __name__ == "__main__":
    main() 