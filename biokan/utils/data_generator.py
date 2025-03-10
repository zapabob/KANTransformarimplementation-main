"""
データ生成ユーティリティ
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_temporal_data(n_samples=1000, sequence_length=20, n_features=64, n_classes=2,
                          temporal_dependency=True, random_state=None):
    """
    時系列データを生成
    
    Args:
        n_samples: サンプル数
        sequence_length: シーケンス長（特徴量の時間的長さ）
        n_features: 特徴数
        n_classes: クラス数
        temporal_dependency: 時間的依存性を有効にするかどうか
        random_state: 乱数シード
    
    Returns:
        X: 特徴量 [n_samples, n_features]
        y: ラベル [n_samples]
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 平坦化前のデータ形状 [n_samples, sequence_length, n_features_per_time]
    n_features_per_time = n_features // sequence_length
    
    # 生データの生成
    X_raw = np.random.randn(n_samples, sequence_length, n_features_per_time)
    
    # 時間的依存性を導入
    if temporal_dependency:
        # 特定の時間ステップに特定のパターンを挿入
        t0 = np.random.randint(0, sequence_length // 3, n_samples)
        t1 = np.random.randint(sequence_length // 2, sequence_length, n_samples)
        
        # ラベル生成（2クラス分類の場合）
        if n_classes == 2:
            # 時間的パターンの存在するサンプルをクラス1に
            y = np.zeros(n_samples, dtype=int)
            
            for i in range(n_samples):
                if np.random.rand() > 0.5:  # 50%の確率でクラス1
                    # 特徴的なパターンを挿入
                    pattern = np.random.randn(n_features_per_time) * 3  # 強いパターン
                    
                    # t0とt1に同じパターンが現れる（時間差パターン）
                    X_raw[i, t0[i], :] += pattern
                    X_raw[i, t1[i], :] += pattern * 0.8  # わずかに変化したパターン
                    
                    y[i] = 1
        else:
            # 多クラス分類の場合
            y = np.random.randint(0, n_classes, n_samples)
            
            for i in range(n_samples):
                cls = y[i]
                
                # クラスごとに異なる時間パターン
                if cls == 0:
                    # パターンなし（そのまま）
                    pass
                elif cls == 1:
                    # t0とt1に正のパターン
                    pattern = np.random.randn(n_features_per_time) * 2 + 1
                    X_raw[i, t0[i], :] += pattern
                    X_raw[i, t1[i], :] += pattern * 0.9
                elif cls == 2:
                    # t0とt1に負のパターン
                    pattern = np.random.randn(n_features_per_time) * 2 - 1
                    X_raw[i, t0[i], :] += pattern
                    X_raw[i, t1[i], :] += pattern * 0.7
                else:
                    # その他のクラスは異なる時間に異なるパターン
                    t2 = (t0[i] + t1[i]) // 2
                    pattern1 = np.random.randn(n_features_per_time) * 2
                    pattern2 = np.random.randn(n_features_per_time) * 2
                    
                    X_raw[i, t0[i], :] += pattern1
                    X_raw[i, t1[i], :] += pattern2
                    X_raw[i, t2, :] += (pattern1 + pattern2) / 2
    else:
        # 時間的依存性なし - 単純な乱数
        y = np.random.randint(0, n_classes, n_samples)
    
    # データを平坦化 [n_samples, sequence_length * n_features_per_time]
    X = X_raw.reshape(n_samples, -1)
    
    # 正規化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def generate_spatial_temporal_data(n_samples=1000, height=32, width=32, n_channels=3, 
                                  n_classes=10, random_state=None):
    """
    空間的・時間的特性を持つデータを生成（画像シーケンスなど）
    
    Args:
        n_samples: サンプル数
        height: 画像の高さ
        width: 画像の幅
        n_channels: チャンネル数
        n_classes: クラス数
        random_state: 乱数シード
    
    Returns:
        X: 特徴量 [n_samples, n_channels, height, width]
        y: ラベル [n_samples]
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 画像データ生成
    X = np.random.randn(n_samples, n_channels, height, width) * 0.1
    
    # ラベル生成
    y = np.random.randint(0, n_classes, n_samples)
    
    # クラスごとの特徴的パターンを追加
    for i in range(n_samples):
        cls = y[i]
        
        # 各クラスに特有のパターンを生成
        if cls == 0:
            # 左上に四角形
            X[i, :, 5:15, 5:15] += 1.0
        elif cls == 1:
            # 右上に四角形
            X[i, :, 5:15, -15:-5] += 1.0
        elif cls == 2:
            # 左下に四角形
            X[i, :, -15:-5, 5:15] += 1.0
        elif cls == 3:
            # 右下に四角形
            X[i, :, -15:-5, -15:-5] += 1.0
        elif cls == 4:
            # 中央に縦線
            X[i, :, 10:-10, 15:17] += 1.5
        elif cls == 5:
            # 中央に横線
            X[i, :, 15:17, 10:-10] += 1.5
        elif cls == 6:
            # 対角線（左上から右下）
            for j in range(min(height, width)):
                if j < height and j < width:
                    X[i, :, j, j] += 2.0
        elif cls == 7:
            # 対角線（右上から左下）
            for j in range(min(height, width)):
                if j < height and j < width:
                    X[i, :, j, width-j-1] += 2.0
        elif cls == 8:
            # 中央に円形
            center_h, center_w = height // 2, width // 2
            radius = min(height, width) // 4
            
            for h in range(height):
                for w in range(width):
                    dist = np.sqrt((h - center_h)**2 + (w - center_w)**2)
                    if dist < radius:
                        X[i, :, h, w] += 1.0
        else:
            # その他のクラスはランダムなノイズを加える
            X[i] += np.random.randn(n_channels, height, width) * 0.5
    
    # ノイズ追加
    X += np.random.randn(n_samples, n_channels, height, width) * 0.2
    
    return X, y 