"""
BioKANモデルの統合訓練スクリプト
基本訓練、転移学習、実験実行の機能を統合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time
from pathlib import Path

# BioKANのインポート
from biokan.core.biokan_model import create_biokan_classifier
from biokan.core.config import BioKANConfig
from biokan.neuro.neuromodulators import NeuromodulatorSystem
from biokan.training.trainer import BioKANTrainer
from biokan.utils.cuda_info_manager import (
    print_cuda_info,
    get_device,
)
from biokan.utils.visualization_utils import setup_japanese_fonts
from biokan.experiments.analyzer import ExperimentAnalyzer


def parse_arguments():
    parser = argparse.ArgumentParser(description="BioKANモデルの統合訓練スクリプト")

    # 基本設定
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "transfer", "experiment"],
        help="実行モード（基本訓練/転移学習/実験）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10", "svhn"],
        help="使用するデータセット",
    )

    # モデル設定
    parser.add_argument("--hidden-dim", type=int, default=128, help="隠れ層の次元数")
    parser.add_argument("--num-blocks", type=int, default=3, help="BioKANブロック数")
    parser.add_argument(
        "--attention-type",
        type=str,
        default="biological",
        choices=["biological", "cortical", "hierarchical"],
        help="アテンションタイプ",
    )

    # 訓練設定
    parser.add_argument("--batch-size", type=int, default=64, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=100, help="エポック数")
    parser.add_argument("--lr", type=float, default=0.001, help="学習率")

    # 転移学習設定
    parser.add_argument("--source-model", type=str, help="転移学習元のモデルパス")
    parser.add_argument("--freeze-layers", type=int, default=2, help="固定する層の数")

    # 実験設定
    parser.add_argument(
        "--experiment-name", type=str, help="実験名（実験モード時に使用）"
    )
    parser.add_argument(
        "--compare-with", type=str, nargs="+", help="比較する実験名のリスト"
    )

    # その他の設定
    parser.add_argument(
        "--save-dir", type=str, default="results", help="結果の保存ディレクトリ"
    )
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")

    return parser.parse_args()


def setup_environment(args):
    # 乱数シードの設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 保存ディレクトリの作成
    save_dir = os.path.join(args.save_dir, args.mode, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # デバイスの設定とCUDA情報の表示
    device = get_device()
    print_cuda_info(verbose=True)

    # 日本語フォントの設定
    setup_japanese_fonts()

    return device, save_dir


def load_dataset(dataset_name, batch_size):
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


def create_model(args, device, in_channels, num_classes):
    if in_channels == 1:
        in_features = 28 * 28  # MNIST/Fashion-MNIST
    else:
        in_features = 32 * 32 * 3  # CIFAR10/SVHN

    model = create_biokan_classifier(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_blocks=args.num_blocks,
        attention_type=args.attention_type,
    )

    if args.mode == "transfer" and args.source_model:
        source_model = create_biokan_classifier(
            in_features=28 * 28,  # MNISTの入力サイズ
            hidden_dim=args.hidden_dim,
            num_classes=10,  # MNISTのクラス数
            num_blocks=args.num_blocks,
            attention_type=args.attention_type,
        )
        source_model.load_state_dict(torch.load(args.source_model, map_location=device))

        # パラメータの転移
        transfer_parameters(model, source_model, args.freeze_layers)

    model = model.to(device)
    return model


def transfer_parameters(target_model, source_model, num_frozen_layers):
    # 共通の層のパラメータを転移
    target_dict = target_model.state_dict()
    source_dict = source_model.state_dict()

    transferred_dict = {}
    for name, param in source_dict.items():
        if name in target_dict and param.size() == target_dict[name].size():
            transferred_dict[name] = param

    target_model.load_state_dict(transferred_dict, strict=False)

    # 指定された層のパラメータを固定
    for i, (name, param) in enumerate(target_model.named_parameters()):
        if i < num_frozen_layers:
            param.requires_grad = False


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # フラット化

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(
                f"エポック: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"損失: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # フラット化
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\nテストセット: 平均損失: {test_loss:.4f}, "
        f"精度: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )

    return test_loss, accuracy


def plot_training_history(
    train_losses, train_accs, test_losses, test_accs, save_dir, mode
):
    plt.figure(figsize=(12, 5))

    # 損失のプロット
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="訓練損失")
    plt.plot(test_losses, label="テスト損失")
    plt.xlabel("エポック")
    plt.ylabel("損失")
    plt.title("訓練・テスト損失の推移")
    plt.legend()

    # 精度のプロット
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="訓練精度")
    plt.plot(test_accs, label="テスト精度")
    plt.xlabel("エポック")
    plt.ylabel("精度 (%)")
    plt.title("訓練・テスト精度の推移")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{mode}_history.png"))
    plt.close()


def run_experiment(args, model, train_loader, test_loader, device, save_dir):
    config = BioKANConfig(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    trainer = BioKANTrainer(model, config)
    trainer.train(train_loader, test_loader, args.epochs)

    if args.compare_with:
        analyzer = ExperimentAnalyzer(save_dir)
        analyzer.compare_learning_curves(
            [args.experiment_name] + args.compare_with, metric="accuracy"
        )
        analyzer.generate_comprehensive_report(
            [args.experiment_name] + args.compare_with,
            os.path.join(save_dir, "analysis_results"),
        )


def main():
    # 引数の解析
    args = parse_arguments()

    # 環境のセットアップ
    device, save_dir = setup_environment(args)

    # データセットの読み込み
    train_loader, test_loader, in_channels, num_classes = load_dataset(
        args.dataset, args.batch_size
    )

    # モデルの作成
    model = create_model(args, device, in_channels, num_classes)

    if args.mode == "experiment":
        # 実験モード
        run_experiment(args, model, train_loader, test_loader, device, save_dir)
    else:
        # 通常の訓練または転移学習モード
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

        # 訓練履歴の保存用リスト
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        best_test_acc = 0

        # 訓練ループ
        print(f"{args.mode}開始: {args.epochs}エポック")
        for epoch in range(1, args.epochs + 1):
            # 訓練
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # 評価
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            # 最良モデルの保存
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"best_{args.mode}_model.pth"),
                )

        # 訓練履歴の可視化
        plot_training_history(
            train_losses, train_accs, test_losses, test_accs, save_dir, args.mode
        )

        print(f"{args.mode}完了！最高テスト精度: {best_test_acc:.2f}%")
        print(f"結果は {save_dir} に保存されました。")


if __name__ == "__main__":
    main()
