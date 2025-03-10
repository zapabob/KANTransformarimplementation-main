"""
BioKANモデルのGPU推論統合スクリプト
CUDA 12対応の最適化と汎用的なGPUサポートを提供
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os
import json
import sys
from PIL import Image


def print_system_info():
    """システム情報の表示"""
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"Python バージョン: {sys.version}")

    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"CUDA バージョン: {cuda_version}")

        # CUDA 12の互換性チェック
        if cuda_version.startswith("12."):
            print("CUDA 12が検出されました。最適化された推論を行います。")
        else:
            print(f"警告: CUDA {cuda_version}が検出されました。")

        # GPU情報
        device_count = torch.cuda.device_count()
        print(f"利用可能なGPUデバイス数: {device_count}")

        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(
                f"GPU {i}: {device_name} (Compute Capability: {device_capability[0]}.{device_capability[1]})"
            )

            # メモリ情報
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  合計メモリ: {total_memory:.2f} GB")
    else:
        print("警告: GPUが検出されませんでした。CPUで実行します。")


def setup_gpu_optimization(cuda12_optimizations=True):
    """GPU最適化の設定"""
    if torch.cuda.is_available():
        # 基本的な最適化
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        if cuda12_optimizations:
            # CUDA 12向けの追加最適化
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("GPU最適化設定が完了しました")
        print("  - CUDNNベンチマークモード: 有効")
        if cuda12_optimizations:
            print("  - TensorCore (TF32): 有効（利用可能な場合）")

        return True
    else:
        print("GPUが利用できないため、最適化は適用されません")
        return False


def run_gpu_inference(
    model,
    data_loader,
    num_samples=10,
    task_type="classification",
    precision="mixed",
    cuda12_optimizations=True,
):
    """
    GPUを使用して高速に推論を実行

    Args:
        model: 推論モデル
        data_loader: データローダー
        num_samples: 推論サンプル数
        task_type: タスクタイプ ('classification' or 'regression')
        precision: 計算精度 ('mixed', 'float32', 'float16', 'bfloat16')
        cuda12_optimizations: CUDA 12向け最適化を有効にするか

    Returns:
        推論結果のリスト
    """
    # GPUの確認と設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPUを使用: {torch.cuda.get_device_name(0)}")
        setup_gpu_optimization(cuda12_optimizations)
    else:
        device = torch.device("cpu")
        print("警告: GPUが利用できないため、CPUを使用します")
        precision = "float32"

    # モデルをGPUに転送
    model = model.to(device)
    model.eval()

    # 精度設定
    print(f"推論精度モード: {precision}")
    if precision == "float16" and device.type == "cuda":
        dtype = torch.float16
        autocast_enabled = False
        print("FP16精度で推論を実行")
        model = model.half()
    elif (
        precision == "bfloat16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        dtype = torch.bfloat16
        autocast_enabled = False
        print("BF16精度で推論を実行")
        model = model.to(dtype=torch.bfloat16)
    elif precision == "mixed" and device.type == "cuda":
        dtype = None
        autocast_enabled = True
        print("混合精度(AMP)で推論を実行")
    else:
        dtype = torch.float32
        autocast_enabled = False
        print("FP32精度で推論を実行")

    results = []
    start_time = time.time()

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if i >= num_samples:
                break

            # データをGPUに転送
            data = data.to(device, dtype=dtype if dtype is not None else None)

            # 推論実行
            if autocast_enabled and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    result = process_inference(model, data, target, task_type)
            else:
                result = process_inference(model, data, target, task_type)

            # GPU同期
            if device.type == "cuda":
                torch.cuda.synchronize()

            results.append(result)

            # 進捗表示
            print(f"サンプル {i+1}/{num_samples} の推論完了")
            if task_type == "classification" and result["confidence"] is not None:
                print(
                    f"  予測: クラス {result['prediction']} (実際: {result['target']}) - "
                    f"信頼度: {result['confidence']*100:.2f}%"
                )
            else:
                print(f"  予測: {result['prediction']:.4f} (実際: {result['target']})")

    # パフォーマンス情報の表示
    end_time = time.time()
    inference_time = end_time - start_time
    print_performance_info(inference_time, num_samples, device)

    # 結果の可視化
    if results:
        visualize_results(results)

    return results


def process_inference(model, data, target, task_type):
    """推論の実行と結果の処理"""
    if hasattr(model, "explain_prediction"):
        # 転移学習モデルの場合
        explanation = model.explain_prediction(data)
        output = explanation["prediction"]

        if task_type == "classification":
            prediction = output[0]
            confidence = explanation["confidence"]
        else:
            prediction = output[0][0]
            confidence = None

        nt_levels = explanation["neurotransmitter_levels"]
    else:
        # 通常のBioKANモデルの場合
        output = model(data)

        if task_type == "classification":
            prediction = torch.argmax(output, dim=1)[0].item()
            probs = F.softmax(output, dim=1)
            confidence = probs[0, prediction].item()
        else:
            prediction = output[0].item()
            confidence = None

        nt_levels = (
            model.get_neuromodulator_levels()
            if hasattr(model, "get_neuromodulator_levels")
            else {}
        )

    return {
        "data": data.cpu().to(torch.float32).numpy(),
        "target": target.item(),
        "prediction": prediction,
        "confidence": confidence,
        "neurotransmitter_levels": nt_levels,
    }


def print_performance_info(inference_time, num_samples, device):
    """パフォーマンス情報の表示"""
    print(f"\n推論完了: {num_samples}サンプル")
    print(f"合計時間: {inference_time:.2f}秒")
    print(f"サンプルあたりの時間: {inference_time/num_samples*1000:.2f}ミリ秒")

    if device.type == "cuda":
        print(
            f"GPU最大メモリ使用量: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB"
        )
        print(f"GPU割り当てキャッシュ: {torch.cuda.memory_reserved()/1024**2:.2f} MB")


def visualize_results(results, save_dir="results/gpu_inference"):
    """結果の可視化"""
    os.makedirs(save_dir, exist_ok=True)

    # 神経伝達物質レベルの可視化
    if results[0]["neurotransmitter_levels"]:
        plt.figure(figsize=(12, 6))
        for i, result in enumerate(results[:5]):
            plt.subplot(1, 5, i + 1)
            nt_levels = result["neurotransmitter_levels"]
            keys = list(nt_levels.keys())
            values = [nt_levels[k] for k in keys]

            bars = plt.bar(keys, values)
            for j, bar in enumerate(bars):
                val = values[j]
                color = (
                    plt.cm.Blues(abs(val / 1.0))
                    if keys[j] == "gaba"
                    else plt.cm.Reds(abs(val / 1.0))
                )
                bar.set_color(color)

            plt.title(f"予測: {result['prediction']}\n実際: {result['target']}")
            plt.xticks(rotation=90)
            plt.ylim(-1.0, 1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "neurotransmitter_levels.png"))
        plt.close()

    # 入力画像と予測結果の可視化
    plt.figure(figsize=(15, 6))
    for i, result in enumerate(results[:5]):
        plt.subplot(1, 5, i + 1)
        img = result["data"].squeeze()
        plt.imshow(img, cmap="gray")
        title = f"予測: {result['prediction']}\n実際: {result['target']}"
        plt.title(
            title, color="green" if result["prediction"] == result["target"] else "red"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="BioKANモデルのGPU推論")
    parser.add_argument(
        "--model-path", type=str, required=True, help="モデルファイルのパス"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="推論するサンプル数"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="タスクの種類",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="mixed",
        choices=["mixed", "float32", "float16", "bfloat16"],
        help="計算精度",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="バッチサイズ")
    parser.add_argument(
        "--cuda12-optimizations",
        action="store_true",
        help="CUDA 12向けの最適化を有効にする",
    )
    args = parser.parse_args()

    # システム情報の表示
    print_system_info()

    try:
        # モデルのロード
        model = torch.load(args.model_path)
        print(f"モデルを読み込みました: {args.model_path}")

        # データローダーの作成
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        test_dataset = datasets.MNIST(
            "data", train=False, download=True, transform=transform
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        # 推論の実行
        results = run_gpu_inference(
            model=model,
            data_loader=test_loader,
            num_samples=args.num_samples,
            task_type=args.task_type,
            precision=args.precision,
            cuda12_optimizations=args.cuda12_optimizations,
        )

        print("推論が正常に完了しました。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
