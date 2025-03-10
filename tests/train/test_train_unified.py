"""
BioKANモデルの統合訓練スクリプトのテストケース
"""

import pytest
import torch
import os
from pathlib import Path
import sys
import torch.nn as nn

# scriptsディレクトリをPYTHONPATHに追加
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))

from train.train_unified import (
    parse_arguments,
    setup_environment,
    load_dataset,
    create_model,
    transfer_parameters,
    train_epoch,
    evaluate,
)


@pytest.fixture
def mock_args():
    class Args:
        mode = "train"
        dataset = "mnist"
        hidden_dim = 64
        num_blocks = 2
        attention_type = "biological"
        batch_size = 32
        epochs = 2
        lr = 0.001
        source_model = None
        freeze_layers = 0
        experiment_name = None
        compare_with = None
        save_dir = "test_results"
        seed = 42

    return Args()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_setup_environment(mock_args, tmp_path):
    mock_args.save_dir = str(tmp_path)
    device, save_dir = setup_environment(mock_args)
    assert isinstance(device, torch.device)
    assert os.path.exists(save_dir)


def test_load_dataset(mock_args):
    train_loader, test_loader, in_channels, num_classes = load_dataset(
        mock_args.dataset, mock_args.batch_size
    )

    assert in_channels == 1  # MNISTの場合
    assert num_classes == 10

    # データローダーの検証
    batch = next(iter(train_loader))
    assert len(batch) == 2  # (data, target)
    assert batch[0].shape[0] == mock_args.batch_size  # バッチサイズ
    assert batch[0].shape[1] == in_channels  # チャネル数


def test_create_model(mock_args, device):
    in_channels = 1
    num_classes = 10
    model = create_model(mock_args, device, in_channels, num_classes)

    # モデルの基本的な検証
    model_device = next(model.parameters()).device
    assert model_device.type == device.type  # デバイスタイプの比較
    assert isinstance(model, nn.Module)  # モデルの型を確認
    assert hasattr(model, 'forward')  # forwardメソッドの存在を確認


def test_transfer_parameters(mock_args, device):
    in_channels = 1
    num_classes = 10

    # ソースモデルとターゲットモデルの作成
    source_model = create_model(mock_args, device, in_channels, num_classes)
    target_model = create_model(mock_args, device, in_channels, num_classes)

    # パラメータ転移の実行
    transfer_parameters(target_model, source_model, num_frozen_layers=2)

    # 固定層のチェック
    frozen_count = 0
    for param in target_model.parameters():
        if not param.requires_grad:
            frozen_count += 1
    assert frozen_count > 0


def test_train_epoch(mock_args, device):
    # モデルとデータの準備
    train_loader, _, in_channels, num_classes = load_dataset(
        mock_args.dataset, mock_args.batch_size
    )
    model = create_model(mock_args, device, in_channels, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=mock_args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 1エポックの訓練
    loss, accuracy = train_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100


def test_evaluate(mock_args, device):
    # モデルとデータの準備
    _, test_loader, in_channels, num_classes = load_dataset(
        mock_args.dataset, mock_args.batch_size
    )
    model = create_model(mock_args, device, in_channels, num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    # 評価の実行
    loss, accuracy = evaluate(model, test_loader, criterion, device)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100


if __name__ == "__main__":
    pytest.main([__file__])
