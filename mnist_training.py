"""
MNISTデータセットを使用したBioKANモデルの学習スクリプト
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from contextlib import nullcontext
from sklearn.metrics import accuracy_score, confusion_matrix

# BioKANモデルのインポート
try:
    from biokan.core.biokan_model import create_biokan_classifier, NeuropharmacologicalBioKAN
    BioKAN_AVAILABLE = True
except ImportError:
    print("BioKANモジュールをインポートできませんでした。シンプルなMLPモデルを使用します。")
    BioKAN_AVAILABLE = False

# CUDA関連の設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 最初のGPUを使用
torch.backends.cudnn.benchmark = True     # CUDNNの自動チューナーを有効化
torch.backends.cudnn.deterministic = True # 再現性のため

# 乱数シードを固定
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
    
    # 訓練データセットの読み込み
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform,
        download=True
    )
    
    # テストデータセットの読み込み
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transform,
        download=True
    )
    
    # データローダーの作成
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"訓練データサイズ: {len(train_dataset):,}件")
    print(f"テストデータサイズ: {len(test_dataset):,}件")
    
    return train_loader, test_loader

# シンプルなMLPクラス（フォールバックとして使用）
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

# BioKANモデルの初期化
def initialize_model():
    print("\nモデルを初期化中...")
    
    # MNISTの入力サイズ: 28x28 = 784
    in_features = 28 * 28
    
    if BioKAN_AVAILABLE:
        try:
            # BioKANモデルを使用
            model = create_biokan_classifier(
                in_features=in_features,
                hidden_dim=128,
                num_classes=10,  # MNIST: 10クラス (0-9)
                num_blocks=3,
                attention_type='standard',  # standardで単純化
                dropout=0.2,
                neuromodulation=True  # 学習時には神経調節を有効化
            )
            print("BioKANモデルを初期化しました")
            return model
        except Exception as e:
            print(f"BioKANモデル初期化エラー: {str(e)}")
            print("標準的なMLPモデルにフォールバック...")
    
    # BioKANが使用できない場合はシンプルなMLPを使用
    print("シンプルなMLPモデルを初期化しました")
    return SimpleMLP(in_features, 128, 10)

# モデルの学習
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, weight_decay=1e-5):
    print("\nモデルの学習を開始...")
    
    # 損失関数と最適化アルゴリズムの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学習率スケジューラ（学習率を徐々に減少させる）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 学習履歴を保存
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'best_test_acc': 0.0,
        'best_epoch': 0
    }
    
    # 開始時間
    start_time = time.time()
    
    # エポックごとの学習ループ
    for epoch in range(epochs):
        model.train()  # 学習モード
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # データローダーからバッチを取得
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"エポック {epoch+1}/{epochs}")):
            # 画像とラベルをデバイスに転送
            images = images.reshape(images.shape[0], -1).to(device)  # [batch, 784]
            labels = labels.to(device)
            
            # 勾配をゼロに初期化
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(images)
            
            # 損失を計算
            loss = criterion(outputs, labels)
            
            # 逆伝播
            loss.backward()
            
            # パラメータを更新
            optimizer.step()
            
            # 統計情報を更新
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()
            
            # GPUメモリをクリア（オプション）
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # エポックの平均損失と精度を計算
        epoch_loss /= len(train_loader)
        epoch_acc = epoch_correct / epoch_total
        
        # テストデータでの評価
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        
        # 学習率を調整
        scheduler.step(test_loss)
        
        # 学習履歴を更新
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 最高精度を保存
        if test_acc > history['best_test_acc']:
            history['best_test_acc'] = test_acc
            history['best_epoch'] = epoch + 1
            
            # 最高精度のモデルを保存
            torch.save(model.state_dict(), 'best_biokan_mnist_model.pth')
            print(f"🔔 新しい最高精度を達成！モデルを保存しました")
        
        # エポックの結果を表示
        print(f"エポック {epoch+1}/{epochs} - "
              f"訓練損失: {epoch_loss:.4f}, 訓練精度: {epoch_acc:.4f}, "
              f"テスト損失: {test_loss:.4f}, テスト精度: {test_acc:.4f}")
    
    # 経過時間
    elapsed_time = time.time() - start_time
    print(f"\n学習完了！総時間: {elapsed_time:.1f}秒")
    print(f"最高テスト精度: {history['best_test_acc']:.4f} (エポック {history['best_epoch']})")
    
    # 最終モデルを保存
    torch.save(model.state_dict(), 'final_biokan_mnist_model.pth')
    print("最終モデルを保存しました: final_biokan_mnist_model.pth")
    
    return model, history

# モデルの評価
def evaluate_model(model, test_loader, criterion=None):
    model.eval()  # 評価モード
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(images.shape[0], -1).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    
    return test_loss, test_acc

# 学習履歴の可視化
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # 損失のプロット
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='訓練損失')
    plt.plot(history['test_loss'], label='テスト損失')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.title('学習損失の推移')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 精度のプロット
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='訓練精度')
    plt.plot(history['test_acc'], label='テスト精度')
    plt.xlabel('エポック')
    plt.ylabel('精度')
    plt.title('学習精度の推移')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('biokan_mnist_training_history.png')
    print("学習履歴グラフを保存しました: biokan_mnist_training_history.png")

# メイン処理
if __name__ == "__main__":
    try:
        # MNISTデータのロード
        train_loader, test_loader = load_mnist_data(batch_size=128)
        
        # モデルの初期化
        model = initialize_model()
        model = model.to(device)
        print(f"モデルを{device}に転送しました")
        
        # モデルの学習
        trained_model, history = train_model(
            model, 
            train_loader, 
            test_loader,
            epochs=15,
            lr=0.001,
            weight_decay=1e-5
        )
        
        # 学習履歴の可視化
        plot_training_history(history)
        
        # 最終評価
        _, final_acc = evaluate_model(trained_model, test_loader)
        print(f"\n最終テスト精度: {final_acc:.4f}")
        
        print("\n学習したモデルは以下のパスに保存されています:")
        print("- 最高精度モデル: best_biokan_mnist_model.pth")
        print("- 最終モデル: final_biokan_mnist_model.pth")
        
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