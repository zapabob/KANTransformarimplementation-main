"""
MNISTデータセットでシンプルなモデルの推論を実行するスクリプト
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

# 必要なディレクトリを作成
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# MNISTデータセットのロード
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# シンプルなニューラルネットワークモデル
class SimpleNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルの作成
model = SimpleNN(hidden_dim=128)
model = model.to(device)
model.eval()  # 評価モード

# サンプルの選択と推論
def run_inference(num_samples=5):
    # 指定されたサンプル数だけテストセットからランダムに選択
    all_indices = list(range(len(test_dataset)))
    np.random.shuffle(all_indices)
    selected_indices = all_indices[:num_samples]
    
    results = []
    
    for idx in selected_indices:
        # サンプルを取得
        x, true_label = test_dataset[idx]
        x = x.to(device)
        x_flat = x.view(1, -1)  # バッチサイズ1、平坦化
        
        # 推論
        with torch.no_grad():
            start_time = time.time()
            outputs = model(x.unsqueeze(0))  # バッチ次元を追加
            inference_time = time.time() - start_time
        
        # 予測結果
        probabilities = F.softmax(outputs, dim=1)
        pred_label = torch.argmax(probabilities, dim=1).item()
        pred_prob = probabilities[0, pred_label].item()
        
        # 結果を保存
        result = {
            'index': idx,
            'true_label': true_label,
            'pred_label': pred_label,
            'correct': (true_label == pred_label),
            'confidence': pred_prob,
            'probabilities': probabilities[0].cpu().numpy(),
            'inference_time': inference_time,
            'input': x.cpu().numpy()
        }
        
        results.append(result)
    
    return results

# 推論の実行
print(f"MNISTデータセットで5サンプルの推論を実行中...")
inference_results = run_inference(num_samples=5)

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
    
    # 結果の表示・保存
    fig = plt.figure(figsize=(15, 5))
    
    for i, result in enumerate(results):
        plt.subplot(1, 5, i+1)
        img = result['input'].squeeze()
        plt.imshow(img, cmap='gray')
        
        title = f"予測: {result['pred_label']}\n正解: {result['true_label']}"
        if result['correct']:
            plt.title(title, color='green')
        else:
            plt.title(title, color='red')
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("results/inference_results.png")
    
    # 予測確率分布のプロット
    plt.figure(figsize=(12, 6))
    
    for i, result in enumerate(results):
        plt.subplot(1, 5, i+1)
        probs = result['probabilities']
        plt.bar(range(10), probs)
        plt.xlabel('数字')
        plt.ylabel('確率')
        plt.title(f"サンプル {i+1}")
        plt.xticks(range(10))
    
    plt.tight_layout()
    plt.savefig("results/prediction_probabilities.png")

# 結果の保存と表示
save_and_display_results(inference_results)

print("\n推論結果を results ディレクトリに保存しました。")
print("推論完了!") 