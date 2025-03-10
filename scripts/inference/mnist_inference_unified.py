"""
BioKANモデルでMNISTデータセットの推論を実行する統合スクリプト
GPU/CPU両対応、神経調節システム対応
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
from biokan.neuro.neuromodulators import NeuromodulatorSystem, PharmacologicalModulator
from biokan.xai.explainers import BioKANExplainer
from biokan.visualization.visualizers import (
    visualize_explanation,
    plot_neurotransmitter_levels,
)
from biokan.utils.cuda_info_manager import (
    print_cuda_info,
    get_device,
    setup_japanese_fonts,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MNISTデータセットでBioKANモデルの推論を実行"
    )
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
    parser.add_argument(
        "--neuromodulation", action="store_true", help="神経調節を有効にする"
    )
    parser.add_argument(
        "--drug_effect",
        type=str,
        default=None,
        choices=[
            "SSRI",
            "SNRI",
            "TCA",
            "typical_antipsychotic",
            "atypical_antipsychotic",
            "benzodiazepine",
            "amphetamine",
            "methylphenidate",
            "caffeine",
        ],
        help="薬理効果をシミュレート",
    )
    parser.add_argument(
        "--drug_dose", type=float, default=0.5, help="薬剤の用量 (0.0-1.0)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="推論するサンプル数"
    )
    parser.add_argument("--explain", action="store_true", help="推論結果の説明を生成")
    parser.add_argument(
        "--save_dir", type=str, default="./results/mnist_inference", help="結果の保存先"
    )
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument(
        "--model_path", type=str, default=None, help="事前学習済みモデルのパス"
    )
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
    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataset, test_loader


def initialize_model(args, device):
    # モデルの作成または読み込み
    model = create_biokan_classifier(
        in_features=28 * 28,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        num_blocks=args.num_blocks,
        attention_type=args.attention_type,
        neuromodulation=args.neuromodulation,
    )

    if args.model_path and os.path.exists(args.model_path):
        print(f"事前学習済みモデルを読み込み: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    model = model.to(device)
    model.eval()
    return model


def run_inference(model, test_dataset, args, device):
    # サンプルの選択
    all_indices = list(range(len(test_dataset)))
    np.random.shuffle(all_indices)
    selected_indices = all_indices[: args.num_samples]

    results = []

    # 神経調節システムの初期化
    if args.neuromodulation:
        neuromodulator_system = NeuromodulatorSystem()
        if args.drug_effect:
            pharmacological_modulator = PharmacologicalModulator(neuromodulator_system)
            print(f"薬理効果 '{args.drug_effect}' を用量 {args.drug_dose} で適用")
            pharmacological_modulator.apply_drug(args.drug_effect, args.drug_dose)

    # 説明器の初期化
    if args.explain:
        explainer = BioKANExplainer(model)

    for idx in selected_indices:
        # サンプルを取得
        x, true_label = test_dataset[idx]
        x = x.to(device)
        x_flat = x.view(1, -1)

        # 推論
        with torch.no_grad():
            start_time = time.time()
            outputs = model(x_flat)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_time = time.time() - start_time

        # 予測結果
        probabilities = F.softmax(outputs, dim=1)
        pred_label = torch.argmax(probabilities, dim=1).item()
        pred_prob = probabilities[0, pred_label].item()

        # 神経伝達物質レベルを取得
        nt_levels = None
        if args.neuromodulation and hasattr(model, "neuromodulator_system"):
            nt_levels = model.neuromodulator_system.get_current_state()

        # 結果を保存
        result = {
            "index": idx,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": (true_label == pred_label),
            "confidence": pred_prob,
            "probabilities": probabilities[0].cpu().numpy(),
            "inference_time": inference_time,
            "input": x.cpu().numpy(),
            "neurotransmitter_levels": nt_levels,
        }

        # 説明の生成
        if args.explain:
            explanation = explainer.analyze(
                x_flat, torch.tensor([true_label]).to(device)
            )
            result["explanation"] = explanation

        results.append(result)

    return results


def save_and_display_results(results, args):
    # 結果サマリー
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    avg_time = np.mean([r["inference_time"] for r in results])

    print(f"\n結果サマリー:")
    print(f"総サンプル数: {len(results)}")
    print(f"正解数: {correct_count}")
    print(f"精度: {accuracy:.4f}")
    print(f"平均推論時間: {avg_time*1000:.2f} ms/サンプル")

    # 推論結果の可視化
    fig = plt.figure(figsize=(15, 10))
    for i, result in enumerate(results):
        plt.subplot(2, 5, i + 1 if i < 10 else 10)
        img = result["input"].squeeze()
        plt.imshow(img, cmap="gray")

        title = f"予測: {result['pred_label']}\n正解: {result['true_label']}"
        plt.title(title, color="green" if result["correct"] else "red")
        plt.axis("off")

        if i >= 9:
            break

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "inference_results.png"))

    # 予測確率分布のプロット
    if results:
        plt.figure(figsize=(12, 6))
        for i, result in enumerate(results[:5]):
            plt.subplot(1, 5, i + 1)
            probs = result["probabilities"]
            plt.bar(range(10), probs)
            plt.xlabel("数字")
            plt.ylabel("確率")
            plt.title(f"サンプル {i+1}")
            plt.xticks(range(10))

        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "prediction_probabilities.png"))

    # 神経伝達物質レベルの可視化
    if args.neuromodulation:
        for i, result in enumerate(results[:5]):
            if result["neurotransmitter_levels"]:
                plt.figure(figsize=(10, 6))
                plot_neurotransmitter_levels(result["neurotransmitter_levels"])
                plt.title(f"サンプル {i+1} の神経伝達物質レベル")
                plt.savefig(
                    os.path.join(args.save_dir, f"neurotransmitters_sample_{i+1}.png")
                )

    # 説明の可視化
    if args.explain:
        for i, result in enumerate(results[:5]):
            if "explanation" in result:
                explanation_dir = os.path.join(
                    args.save_dir, f"explanation_sample_{i+1}"
                )
                os.makedirs(explanation_dir, exist_ok=True)
                explanation_figs = visualize_explanation(result["explanation"])
                for name, fig in explanation_figs.items():
                    plt.figure(fig.number)
                    plt.savefig(os.path.join(explanation_dir, f"{name}.png"))
                    plt.close(fig)


def main():
    # 引数の解析
    args = parse_arguments()

    # 環境のセットアップ
    device = setup_environment(args)

    # データの読み込み
    test_dataset, test_loader = load_data(args.batch_size)

    # モデルの初期化
    model = initialize_model(args, device)

    # 推論の実行
    print(f"MNISTデータセットで{args.num_samples}サンプルの推論を実行中...")
    inference_results = run_inference(model, test_dataset, args, device)

    # 結果の保存と表示
    save_and_display_results(inference_results, args)

    print(f"\n推論結果を {args.save_dir} に保存しました。")
    print("推論完了!")


if __name__ == "__main__":
    main()
