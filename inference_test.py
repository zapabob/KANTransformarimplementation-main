"""
BioKANモデルの推論テストと決定係数の計算
（単純化バージョン - 基本的なMLPに変更）
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
import os
from contextlib import nullcontext

# CUDA関連の設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 最初のGPUを使用
torch.backends.cudnn.benchmark = True      # CUDNNの自動チューナーを有効化
torch.backends.cudnn.deterministic = True  # 再現性のため

# 乱数シードを固定
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# シンプルなMLPモデル定義
class SimpleRegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=3):
        super().__init__()
        
        layers = []
        current_dim = in_features
        
        # 隠れ層の構築
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        # 出力層
        layers.append(nn.Linear(current_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# テストデータの生成
def generate_test_data(n_samples=1000, seq_length=4, feature_dim=16):
    print("テストデータを生成中...")
    X = np.random.randn(n_samples, seq_length, feature_dim)  # (batch_size, seq_length, features)
    # 非線形な関係性を持つ目的変数を生成
    y = 0.3 * np.sin(X[:, 0, 0]) + 0.7 * np.exp(-X[:, 0, 1]**2) + 0.2 * X[:, 0, 2]**2
    return X, y

# GPU利用可能性の確認
if not torch.cuda.is_available():
    print("警告: GPUが利用できません。CPUを使用します。")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"利用可能なGPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# テストデータの準備
print("\nデータを準備中...")
X_test, y_test = generate_test_data()
# シーケンスと特徴量を結合して平坦化
X_test_flattened = X_test.reshape(X_test.shape[0], -1)  
feature_dim = X_test_flattened.shape[1]  # 平坦化後の特徴量の次元

print(f"元の入力データ形状: {X_test.shape}")
print(f"平坦化後の形状: {X_test_flattened.shape}")

# モデルの初期化
print("\nシンプルなMLPモデルを初期化中...")
model = SimpleRegressor(
    in_features=feature_dim,  # 入力特徴量の次元（平坦化後）
    hidden_dim=64,            # 隠れ層の次元
    num_layers=3              # レイヤー数
)

# モデルをGPUに転送
model = model.to(device)
print(f"モデルを{device}に転送しました")

# テストデータをPyTorchテンソルに変換
X_test_tensor = torch.FloatTensor(X_test_flattened).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# バッチサイズの設定
batch_size = 32  # バッチサイズ
n_batches = len(X_test) // batch_size + (1 if len(X_test) % batch_size != 0 else 0)

print(f"バッチサイズ: {batch_size}")
print(f"バッチ数: {n_batches}")

# 推論の実行
print("\n推論を実行中...")
model.eval()
predictions = []

try:
    with torch.no_grad():
        # torch.cuda.amp.autocastは非推奨なのでtorch.amp.autocastを使用
        amp_context = torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext()
        with amp_context:
            for i in tqdm(range(0, len(X_test), batch_size), desc="推論進捗"):
                batch_x = X_test_tensor[i:i+batch_size]
                batch_pred = model(batch_x).squeeze()
                predictions.append(batch_pred.cpu().numpy())
                if i == 0:
                    print(f"予測結果形状: {batch_pred.shape}")

    # numpy配列に変換
    if predictions:
        if isinstance(predictions[0], np.ndarray) and len(predictions[0].shape) == 0:
            y_pred = np.array([p.item() if hasattr(p, 'item') else p for p in predictions])
        else:
            try:
                y_pred = np.concatenate(predictions)
            except ValueError as e:
                print(f"予測結果の連結でエラー: {e}")
                # 形状を表示
                for i, pred in enumerate(predictions):
                    print(f"予測{i}の形状: {pred.shape}")
                # すべての予測を1次元にする
                y_pred = np.array([p.mean() if hasattr(p, 'mean') else p for p in predictions])
    else:
        print("警告: 予測結果がありません")
        y_pred = np.array([])

    # 決定係数（R²）の計算
    if len(y_pred) > 0 and len(y_pred) == len(y_test):
        r2 = r2_score(y_test[:len(y_pred)], y_pred)
        
        print("\n" + "="*50)
        print("MLPモデル推論結果")
        print("="*50)
        print(f"テストデータサイズ: {len(X_test):,}件")
        print(f"入力特徴量次元: {feature_dim}次元")
        print(f"バッチサイズ: {batch_size}")
        print(f"決定係数 (R²): {r2:.4f}")
        print("\n予測値の統計:")
        print(f"平均: {np.mean(y_pred):.4f}")
        print(f"標準偏差: {np.std(y_pred):.4f}")
        print(f"最小値: {np.min(y_pred):.4f}")
        print(f"最大値: {np.max(y_pred):.4f}")
    else:
        print(f"警告: 予測値数({len(y_pred)})と正解値数({len(y_test)})が一致しません")

except Exception as e:
    print(f"\nエラーが発生しました: {str(e)}")
    print("\nGPUメモリ使用状況:")
    if torch.cuda.is_available():
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