"""
BioKANモデルでMNISTデータセットの推論を実行するスクリプト
（GPU/CPU両対応版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import shutil
from cuda_info_manager import print_cuda_info, get_device, setup_japanese_fonts

# 日本語フォントの設定（初回のみ詳細表示）
setup_japanese_fonts(verbose=True)

# 必要なディレクトリを作成
os.makedirs('data', exist_ok=True)
os.makedirs('biokan_results', exist_ok=True)

# BioKANモデルクラスを定義
class BioKANModel(nn.Module):
    def __init__(self, in_features=784, hidden_dim=128, num_classes=10, num_blocks=2):
        super(BioKANModel, self).__init__()
        self.flatten = nn.Flatten()
        
        # 入力層
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # BioKANブロック（複数層のニューラルネットワーク）
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.blocks.append(block)
        
        # 出力層
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        
        # 神経調節システムのシミュレーション
        self.neurotransmitter_levels = {
            'dopamine': 0.2,    # 報酬シグナル
            'serotonin': 0.3,   # 感情調整
            'noradrenaline': 0.4,  # 覚醒・注意
            'acetylcholine': 0.5,  # 記憶と学習
            'glutamate': 0.6,   # 興奮性信号
            'gaba': -0.3        # 抑制性信号
        }
    
    def forward(self, x):
        # 入力データをフラット化
        x = self.flatten(x)
        
        # 入力投影
        x = F.relu(self.input_proj(x))
        
        # BioKANブロックを通過
        for block in self.blocks:
            # 神経調節の影響をシミュレート
            attention_factor = 1.0 + 0.2 * self.neurotransmitter_levels['noradrenaline']
            learning_factor = 1.0 + 0.3 * self.neurotransmitter_levels['acetylcholine']
            
            # ブロック出力に調節効果を適用
            block_output = block(x)
            x = x + block_output * attention_factor * learning_factor
        
        # 出力投影
        x = self.output_proj(x)
        return x
    
    def get_neurotransmitter_levels(self):
        return self.neurotransmitter_levels

# CUDAが利用可能か確認
try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDAデバイスを使用: {torch.cuda.get_device_name(0)}")
        print(f"利用可能なGPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDAデバイスが見つかりません。CPUを使用します。")
        device = torch.device('cpu')
except Exception as e:
    print(f"CUDA初期化エラー: {e}")
    print("CPUを使用します。")
    device = torch.device('cpu')

# MNISTデータセットのロード
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# BioKANモデルの作成
model = BioKANModel(in_features=28*28, hidden_dim=128, num_classes=10, num_blocks=3)
model = model.to(device)
model.eval()  # 評価モード

# サンプルの選択と推論
def run_inference(num_samples=10):
    """
    BioKANモデルを使用してMNISTデータセットの推論を実行
    
    Args:
        num_samples: テストセットから選択するサンプル数
        
    Returns:
        結果の辞書（予測ラベル、信頼度など）
    """
    # 推論時間の計測開始
    start_time = time.time()
    
    # デバイスの設定（GPU/CPU）
    try:
        device = get_device()
        print_cuda_info(verbose=True)
    except Exception as e:
        print(f"デバイス初期化エラー: {e}")
        device = torch.device('cpu')
    
    # MNISTデータセットのロード
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # テストデータセットの準備
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # 指定されたサンプル数だけテストセットからランダムに選択
    all_indices = list(range(len(test_dataset)))
    selected_indices = np.random.choice(all_indices, num_samples, replace=False)
    
    # モデルのロード
    model = BioKANModel().to(device)
    print(f"BioKANモデルでMNISTデータセット{num_samples}サンプルの推論を実行中...")
    
    # 結果の格納用リスト
    results = []
    
    # 正解数のカウント
    correct_count = 0
    total_inference_time = 0
    
    # 選択したサンプルで推論を実行
    for i, idx in enumerate(selected_indices):
        # サンプルの取得
        x, y = test_dataset[idx]
        x = x.to(device)
        true_label = y
        
        # 推論
        with torch.no_grad():
            start_time = time.time()
            # バッチ次元を追加してモデルに入力
            outputs = model(x.unsqueeze(0))
            # CUDA同期を確実に行う（CUDAが利用可能な場合のみ）
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start_time
        
        # 予測結果
        probabilities = F.softmax(outputs, dim=1)
        pred_label = torch.argmax(probabilities, dim=1).item()
        pred_prob = probabilities[0, pred_label].item()
        
        # 神経伝達物質レベルを取得
        nt_levels = model.get_neurotransmitter_levels()
        
        # 結果を保存
        result = {
            'index': idx,
            'input': x.cpu().numpy(),
            'true_label': true_label,
            'pred_label': pred_label,
            'correct': pred_label == true_label,
            'confidence': pred_prob,
            'probabilities': probabilities[0].cpu().numpy(),
            'inference_time': inference_time,
            'neurotransmitter_levels': nt_levels
        }
        
        results.append(result)
        
        # 正解数をカウント
        if pred_label == true_label:
            correct_count += 1
            
        total_inference_time += inference_time
        
    # 平均推論時間を計算
    avg_time = total_inference_time / num_samples
    
    # 結果の要約表示
    print(f"\n総サンプル数: {num_samples}")
    print(f"正解数: {correct_count}")
    print(f"精度: {correct_count/num_samples:.4f}")
    print(f"平均推論時間: {avg_time*1000:.2f} ms/サンプル")
    print(f"使用デバイス: {device.type}")
    
    # 結果の表示・保存
    fig = plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        plt.subplot(2, 5, i+1)
        img = result['input'].squeeze()
        plt.imshow(img, cmap='gray')
        
        title = f"Pred/予測: {result['pred_label']}\nTrue/正解: {result['true_label']}"
        if result['correct']:
            plt.title(title, color='green')
        else:
            plt.title(title, color='red')
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("biokan_results/inference_results.png")
    
    # 予測確率分布のプロット
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        plt.subplot(2, 5, i+1)
        probs = result['probabilities']
        plt.bar(range(10), probs)
        plt.xlabel('Digit/数字')
        plt.ylabel('Probability/確率')
        plt.title(f"Sample/サンプル {i+1}")
        plt.xticks(range(10))
    
    plt.tight_layout()
    plt.savefig("biokan_results/prediction_probabilities.png")
    
    # 神経伝達物質レベルの可視化
    plt.figure(figsize=(10, 6))
    
    nt_levels = results[0]['neurotransmitter_levels']  # すべてのサンプルで同じ
    labels = list(nt_levels.keys())
    values = list(nt_levels.values())
    
    plt.bar(labels, values)
    plt.title("Neurotransmitter Levels / 神経伝達物質レベル")
    plt.ylabel("Level / レベル")
    plt.tight_layout()
    plt.savefig("biokan_results/neurotransmitters.png")
    
    # GPUメモリ使用状況の表示（GPUが利用可能な場合のみ）
    if device.type == 'cuda':
        try:
            print("\nGPUメモリ使用量:")
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"割り当て済み: {allocated:.2f} MB")
            print(f"キャッシュ: {reserved:.2f} MB")
            
            # メモリのクリーンアップ
            torch.cuda.empty_cache()
            
            print("メモリクリーンアップ後:")
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"割り当て済み: {allocated:.2f} MB")
            print(f"キャッシュ: {reserved:.2f} MB")
        except Exception as e:
            print(f"GPUメモリ情報取得エラー: {e}")
    
    print("\n推論結果を biokan_results ディレクトリに保存しました。")
    print("BioKANモデルによる推論完了!")
    
    return results

# 推論の実行
print(f"BioKANモデルでMNISTデータセット10サンプルの推論を実行中...")
inference_results = run_inference(num_samples=10)

# 結果の保存と表示
def save_and_display_results(results):
    # 結果サマリー
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results)
    avg_time = np.mean([r['inference_time'] for r in results])
    
    print(f"\n結果サマリー:")
    print(f"総サンプル数: {len(results)}")
    print(f"正解数: {correct_count}")
    print(f"精度: {accuracy:.4f}")
    print(f"平均推論時間: {avg_time*1000:.2f} ms/サンプル")
    print(f"使用デバイス: {device.type}")
    
    # 結果の表示・保存
    fig = plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        plt.subplot(2, 5, i+1)
        img = result['input'].squeeze()
        plt.imshow(img, cmap='gray')
        
        title = f"Pred/予測: {result['pred_label']}\nTrue/正解: {result['true_label']}"
        if result['correct']:
            plt.title(title, color='green')
        else:
            plt.title(title, color='red')
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("biokan_results/inference_results.png")
    
    # 予測確率分布のプロット
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        plt.subplot(2, 5, i+1)
        probs = result['probabilities']
        plt.bar(range(10), probs)
        plt.xlabel('Digit/数字')
        plt.ylabel('Probability/確率')
        plt.title(f"Sample/サンプル {i+1}")
        plt.xticks(range(10))
    
    plt.tight_layout()
    plt.savefig("biokan_results/prediction_probabilities.png")
    
    # 神経伝達物質レベルの可視化
    plt.figure(figsize=(10, 6))
    
    nt_levels = results[0]['neurotransmitter_levels']  # すべてのサンプルで同じ
    labels = list(nt_levels.keys())
    values = list(nt_levels.values())
    
    plt.bar(labels, values)
    plt.title("Neurotransmitter Levels / 神経伝達物質レベル")
    plt.ylabel("Level / レベル")
    plt.tight_layout()
    plt.savefig("biokan_results/neurotransmitters.png")

# 結果の保存と表示
save_and_display_results(inference_results)

# GPUメモリ使用状況の出力（CUDAが利用可能な場合のみ）
if device.type == 'cuda':
    try:
        print(f"\nGPUメモリ使用量:")
        print(f"割り当て済み: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"キャッシュ: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
        
        # GPUメモリのクリーンアップ
        torch.cuda.empty_cache()
        print(f"メモリクリーンアップ後:")
        print(f"割り当て済み: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"キャッシュ: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    except Exception as e:
        print(f"GPUメモリ情報取得エラー: {e}")

print("\n推論結果を biokan_results ディレクトリに保存しました。")
print("BioKANモデルによる推論完了!") 