"""
BioKANモデルのMNIST訓練統合スクリプト
基本訓練と転移学習の両方をサポート
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
from biokan.neuro.neuromodulators import NeuromodulatorSystem
from biokan.training.trainer import BioKANTrainer
from biokan.utils.cuda_info_manager import (
    print_cuda_info,
    get_device,
    setup_japanese_fonts,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="BioKANモデルのMNIST訓練")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隠れ層の次元数")
    parser.add_argument("--num_blocks", type=int, default=3, help="BioKANブロック数")
    parser.add_argument(
        "--attention_type",
        type=str,
        default="biological",
        choices=["biological", "cortical", "hierarchical"],
        help="アテンションタイプ",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=100, help="エポック数")
    parser.add_argument("--lr", type=float, default=0.001, help="学習率")
    parser.add_argument(
        "--neuromodulation", action="store_true", help="神経調節を有効にする"
    )
    parser.add_argument(
        "--transfer_learning", action="store_true", help="転移学習を有効にする"
    )
    parser.add_argument(
        "--source_model", type=str, default=None, help="転移学習元のモデルパス"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mnist_training",
        help="結果の保存ディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    return parser.parse_args()


def setup_environment(args):
    # 乱数シードの設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 保存ディレクトリの作成
    os.makedirs(args.save_dir, exist_ok=True)

    # デバイスの設定とCUDA情報の表示
    device = get_device()
    print_cuda_info(verbose=True)

    # 日本語フォントの設定
    setup_japanese_fonts()

    return device


def load_data(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def initialize_model(args, device):
    # モデルの作成
    model = create_biokan_classifier(
        in_features=28 * 28,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        num_blocks=args.num_blocks,
        attention_type=args.attention_type,
        neuromodulation=args.neuromodulation,
    )

    if args.transfer_learning and args.source_model:
        print(f"転移学習元モデルを読み込み: {args.source_model}")
        source_state = torch.load(args.source_model, map_location=device)
        # 転移可能な層のパラメータをコピー
        model.load_transfer_parameters(source_state)

    model = model.to(device)
    return model


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


def plot_training_history(train_losses, train_accs, test_losses, test_accs, save_dir):
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
    plt.savefig(os.path.join(save_dir, "training_history.png"))
    plt.close()


def main():
    # 引数の解析
    args = parse_arguments()

    # 環境のセットアップ
    device = setup_environment(args)

    # データの読み込み
    train_loader, test_loader = load_data(args.batch_size)

    # モデルの初期化
    model = initialize_model(args, device)

    # 損失関数と最適化器の設定
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 訓練履歴の保存用リスト
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    best_test_acc = 0

    # 訓練ループ
    print(f"訓練開始: {args.epochs}エポック")
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
                model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
            )

    # 訓練履歴の可視化
    plot_training_history(
        train_losses, train_accs, test_losses, test_accs, args.save_dir
    )

    print(f"訓練完了！最高テスト精度: {best_test_acc:.2f}%")
    print(f"結果は {args.save_dir} に保存されました。")


if __name__ == "__main__":
    main()
