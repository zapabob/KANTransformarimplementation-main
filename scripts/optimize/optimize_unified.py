"""
BioKANモデルのハイパーパラメータ最適化統合スクリプト
Optunaを使用した最適化をサポート
"""

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
from pathlib import Path

from biokan.core.biokan_model import create_biokan_classifier
from biokan.core.config import BioKANConfig
from biokan.training.trainer import BioKANTrainer
from biokan.utils.cuda_info_manager import (
    print_cuda_info,
    get_device,
)
from biokan.utils.visualization_utils import setup_japanese_fonts


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="BioKANモデルのハイパーパラメータ最適化"
    )

    # 最適化設定
    parser.add_argument("--n-trials", type=int, default=100, help="最適化試行回数")
    parser.add_argument(
        "--study-name", type=str, default="biokan_optimization", help="Optunaスタディ名"
    )
    parser.add_argument(
        "--storage", type=str, help="Optuna用のストレージURL（SQLite等）"
    )

    # データセット設定
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10", "svhn"],
        help="使用するデータセット",
    )

    # 訓練設定
    parser.add_argument("--epochs", type=int, default=50, help="各試行の訓練エポック数")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/optimization",
        help="結果の保存ディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")

    return parser.parse_args()


def setup_environment(args):
    # 乱数シードの設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 保存ディレクトリの作成
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # デバイスの設定とCUDA情報の表示
    device = get_device()
    print_cuda_info(verbose=True)

    # 日本語フォントの設定
    setup_japanese_fonts()

    return device, save_dir


def load_dataset(dataset_name, batch_size):
    from torchvision import datasets, transforms

    dataset_configs = {
        "mnist": {
            "transform": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
            "dataset_class": datasets.MNIST,
            "in_channels": 1,
            "num_classes": 10,
        },
        "fashion_mnist": {
            "transform": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
            ),
            "dataset_class": datasets.FashionMNIST,
            "in_channels": 1,
            "num_classes": 10,
        },
        "cifar10": {
            "transform": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
            "dataset_class": datasets.CIFAR10,
            "in_channels": 3,
            "num_classes": 10,
        },
        "svhn": {
            "transform": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
                    ),
                ]
            ),
            "dataset_class": datasets.SVHN,
            "in_channels": 3,
            "num_classes": 10,
        },
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"未対応のデータセット: {dataset_name}")

    config = dataset_configs[dataset_name]

    if dataset_name == "svhn":
        train_dataset = config["dataset_class"](
            "data", split="train", download=True, transform=config["transform"]
        )
        test_dataset = config["dataset_class"](
            "data", split="test", transform=config["transform"]
        )
    else:
        train_dataset = config["dataset_class"](
            "data", train=True, download=True, transform=config["transform"]
        )
        test_dataset = config["dataset_class"](
            "data", train=False, transform=config["transform"]
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, config["in_channels"], config["num_classes"]


def create_model(trial, in_channels, num_classes, device):
    """モデルを作成する関数

    Args:
        trial: Optunaのtrial
        in_channels (int): 入力チャネル数
        num_classes (int): クラス数
        device (str): デバイス

    Returns:
        tuple: (model, optimizer)
    """
    # ハイパーパラメータの設定
    num_heads = trial.suggest_int("num_heads", 4, 8)
    hidden_dim_base = trial.suggest_int("hidden_dim_base", 16, 64)
    hidden_dim = hidden_dim_base * num_heads  # num_headsの倍数になるように調整
    num_blocks = trial.suggest_int("num_blocks", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # モデルの作成
    model = create_biokan_classifier(
        in_features=in_channels * 28 * 28,  # MNISTの場合
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=dropout,
        device=device
    )
    model = model.to(device)

    # オプティマイザの設定
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer


def objective(trial, args, device, train_loader, test_loader, in_channels, num_classes):
    # モデルとオプティマイザの作成
    model, optimizer = create_model(trial, in_channels, num_classes, device)
    criterion = nn.CrossEntropyLoss()

    # 訓練ループ
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # フラット化

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 中間評価
            if batch_idx % 100 == 0:
                intermediate_value = evaluate(model, test_loader, device)
                trial.report(intermediate_value, epoch * len(train_loader) + batch_idx)

                if trial.should_prune():
                    raise optuna.TrialPruned()

    # 最終評価
    final_accuracy = evaluate(model, test_loader, device)
    return final_accuracy


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # フラット化
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def save_best_params(study, save_dir):
    best_params = study.best_params
    best_value = study.best_value

    result = {
        "best_params": best_params,
        "best_accuracy": best_value,
        "n_trials": len(study.trials),
        "study_name": study.study_name,
    }

    save_path = os.path.join(save_dir, "best_params.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n最適化結果:")
    print(f"最良の精度: {best_value:.2f}%")
    print(f"最適なパラメータ:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\n結果は {save_path} に保存されました。")


def main():
    # 引数の解析
    args = parse_arguments()

    # 環境のセットアップ
    device, save_dir = setup_environment(args)

    # データセットの読み込み
    train_loader, test_loader, in_channels, num_classes = load_dataset(
        args.dataset, batch_size=64
    )  # バッチサイズは固定

    # Optunaスタディの作成
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    # 最適化の実行
    study.optimize(
        lambda trial: objective(
            trial, args, device, train_loader, test_loader, in_channels, num_classes
        ),
        n_trials=args.n_trials,
    )

    # 最適なパラメータの保存
    save_best_params(study, save_dir)


if __name__ == "__main__":
    main()
