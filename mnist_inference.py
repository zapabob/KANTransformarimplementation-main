"""
MNISTデータセットに対する推論テスト
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from contextlib import nullcontext
from sklearn.metrics import accuracy_score, confusion_matrix

# CUDA関連の設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 最初のGPUを使用
torch.backends.cudnn.benchmark = True     # CUDNNの自動チューナーを有効化
torch.backends.cudnn.deterministic = True # 再現性のため

# 乱数シードを固定
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# GPU利用可能性の確認
if not torch.cuda.is_available():
    print("警告: GPUが利用できません。CPUを使用します。")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"利用可能なGPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# MNISTデータセットの読み込み
def load_mnist_data(batch_size=64):
    print("\nMNISTデータセットを読み込み中...")
    
    # 変換の定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNISTの平均と標準偏差
    ])
    
    # テストデータセットの読み込み
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transform,
        download=True
    )
    
    # データローダーの作成
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"テストデータサイズ: {len(test_dataset):,}件")
    return test_loader

# シンプルなMLPクラス
class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# モデルの初期化
def initialize_model():
    print("\nモデルを初期化中...")
    
    # MNISTの入力サイズ: 28x28 = 784
    in_features = 28 * 28
    
    # シンプルなMLPモデルを使用
    return SimpleMLP(in_features, 128, 10)

# モデルの推論実行
def run_inference(model, test_loader):
    print("\nMNISTデータに対して推論を実行中...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # torch.cuda.amp.autocastは非推奨なのでtorch.amp.autocastを使用
        amp_context = torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext()
        with amp_context:
            for images, labels in tqdm(test_loader, desc="推論進捗"):
                # MNISTは[batch, 1, 28, 28]形式なので[batch, 784]に変換
                images = images.reshape(images.shape[0], -1).to(device)
                labels = labels.to(device)
                
                # 推論
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                # 結果を保存
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    
    # 予測結果を連結
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return all_preds, all_labels

# 結果の評価と可視化
def evaluate_results(predictions, labels):
    print("\n" + "="*50)
    print("MNIST推論結果")
    print("="*50)
    
    # 精度の計算
    accuracy = accuracy_score(labels, predictions)
    print(f"精度: {accuracy:.4f}")
    
    # 混同行列の計算
    cm = confusion_matrix(labels, predictions)
    
    # 混同行列の可視化
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("混同行列")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, np.arange(10))
    plt.yticks(tick_marks, np.arange(10))
    plt.xlabel("予測ラベル")
    plt.ylabel("実際のラベル")
    
    # 各セルの値を表示
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig("mnist_confusion_matrix.png")
    print("混同行列を保存しました: mnist_confusion_matrix.png")
    
    # クラスごとの精度
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(10), class_accuracy)
    plt.xlabel("数字")
    plt.ylabel("精度")
    plt.title("クラスごとの精度")
    plt.xticks(np.arange(10))
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("mnist_class_accuracy.png")
    print("クラス別精度を保存しました: mnist_class_accuracy.png")
    
    return accuracy

# メイン処理
if __name__ == "__main__":
    try:
        # MNISTデータの読み込み
        test_loader = load_mnist_data(batch_size=128)
        
        # モデルの初期化
        model = initialize_model()
        model = model.to(device)
        print(f"モデルを{device}に転送しました")
        
        # 事前学習済みモデルの読み込み（存在する場合）
        model_path = "mnist_model.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"事前学習済みモデルを読み込みました: {model_path}")
            except Exception as e:
                print(f"モデル読み込みエラー: {str(e)}")
                print("事前学習なしで推論を実行します")
        else:
            print("事前学習済みモデルが見つかりません。ランダム初期化で推論を実行します。")
            print("注: 事前学習なしの場合、精度はランダム推測（約10%）と同程度になります。")
        
        # 推論実行
        predictions, labels = run_inference(model, test_loader)
        
        # 結果評価
        accuracy = evaluate_results(predictions, labels)
        
        print("\n" + "="*50)
        print(f"最終精度: {accuracy:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        # GPUメモリ使用状況の表示
        if torch.cuda.is_available():
            print("\nGPUメモリ使用状況:")
            print(f"割り当て済みメモリ: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"キャッシュされたメモリ: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
            # エラーの詳細情報を出力
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()  # GPUメモリの解放
    
    finally:
        # クリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 