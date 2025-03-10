"""
BioKANモデルの最適化スクリプトのテストケース
"""

import pytest
import torch
import os
from pathlib import Path
import sys
import optuna
import torch.nn as nn

# scriptsディレクトリをPYTHONPATHに追加
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))

from optimize.optimize_unified import (
    parse_arguments,
    setup_environment,
    load_dataset,
    create_model,
    objective,
    evaluate,
    save_best_params,
)


@pytest.fixture
def mock_args():
    class Args:
        n_trials = 2
        study_name = "test_optimization"
        storage = None
        dataset = "mnist"
        epochs = 1
        save_dir = "test_results"
        seed = 42

    return Args()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_trial():
    study = optuna.create_study(direction="maximize")
    return study.ask()


def test_setup_environment(mock_args, tmp_path):
    mock_args.save_dir = str(tmp_path)
    device, save_dir = setup_environment(mock_args)
    assert isinstance(device, torch.device)
    assert os.path.exists(save_dir)


def test_load_dataset(mock_args):
    train_loader, test_loader, in_channels, num_classes = load_dataset(
        mock_args.dataset, batch_size=32
    )

    assert in_channels == 1  # MNISTの場合
    assert num_classes == 10

    # データローダーの検証
    batch = next(iter(train_loader))
    assert len(batch) == 2  # (data, target)
    assert batch[0].shape[1] == in_channels  # チャネル数


def test_create_model(mock_trial, device):
    in_channels = 1
    num_classes = 10
    model, optimizer = create_model(mock_trial, in_channels, num_classes, device)

    # モデルの基本的な検証
    model_device = next(model.parameters()).device
    assert model_device.type == device.type  # デバイスタイプの比較
    assert isinstance(model, nn.Module)  # モデルの型を確認
    assert hasattr(model, 'forward')  # forwardメソッドの存在を確認

    # オプティマイザの検証
    assert isinstance(optimizer, torch.optim.Adam)

    # 順伝播のテスト
    batch_size = 2
    x = torch.randn(batch_size, 28 * 28).to(device)
    output = model(x)
    assert output.shape == (batch_size, num_classes)


def test_evaluate(mock_args, mock_trial, device):
    # モデルとデータの準備
    _, test_loader, in_channels, num_classes = load_dataset(
        mock_args.dataset, batch_size=32
    )
    model, _ = create_model(mock_trial, in_channels, num_classes, device)

    # 評価の実行
    accuracy = evaluate(model, test_loader, device)

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100


def test_objective(mock_args, mock_trial, device):
    # データの準備
    train_loader, test_loader, in_channels, num_classes = load_dataset(
        mock_args.dataset, batch_size=32
    )

    # 目的関数の実行
    accuracy = objective(
        mock_trial,
        mock_args,
        device,
        train_loader,
        test_loader,
        in_channels,
        num_classes,
    )

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100


def test_save_best_params(tmp_path):
    # モックのスタディを作成
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: t.suggest_float("x", -10, 10), n_trials=1)

    # 結果の保存
    save_dir = tmp_path
    save_best_params(study, save_dir)

    # 保存されたファイルの検証
    result_file = os.path.join(save_dir, "best_params.json")
    assert os.path.exists(result_file)


if __name__ == "__main__":
    pytest.main([__file__])
