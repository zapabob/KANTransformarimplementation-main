"""
NeoCortexBioKAN - 人間の大脳新皮質の機能を忠実に模倣するニューラルネットワークモデル

参考文献:
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
- Sporns, O. (2011). Networks of the Brain. MIT Press.
- Buzsáki, G. (2006). Rhythms of the Brain. Oxford University Press.
- Goldman-Rakic, P.S. (1995). Cellular basis of working memory. Neuron, 14(3), 477-485.
- Lake, B.M., et al. (2017). Building machines that learn and think like people. Behavioral and Brain Sciences, 40, e253.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm

# プロジェクトルートへのパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from biokan.core.biokan_model import NeoCortexBioKAN
from biokan.utils.visualization import visualize_layer_activations, plot_astrocyte_calcium

def generate_complex_sequence_data(n_samples=1000, seq_length=32, n_features=10, pattern_complexity=3):
    """
    より複雑なパターンを持つシーケンスデータを生成
    （人間の認知パターンを模倣）
    
    Args:
        n_samples: サンプル数
        seq_length: シーケンス長
        n_features: 特徴量の次元
        pattern_complexity: パターンの複雑さ
        
    Returns:
        X: 入力データ
        y: ターゲットラベル
    """
    X = np.zeros((n_samples, seq_length, n_features))
    y = np.zeros(n_samples, dtype=np.int64)
    
    # 異なるクラスのパターンを生成
    patterns = []
    for i in range(10):  # 10クラス
        pattern = np.random.randn(pattern_complexity, n_features) * 0.5
        patterns.append(pattern)
    
    for i in range(n_samples):
        # クラスをランダムに選択
        cls = np.random.randint(0, 10)
        y[i] = cls
        
        # 基本的なノイズ
        X[i] = np.random.randn(seq_length, n_features) * 0.1
        
        # パターンの配置（不規則な間隔で繰り返し）
        for j in range(0, seq_length, pattern_complexity + np.random.randint(1, 4)):
            if j + pattern_complexity <= seq_length:
                X[i, j:j+pattern_complexity] += patterns[cls]
        
        # 長期依存性を追加
        if seq_length > 20:
            # 序盤のパターンと終盤のパターンを関連づける
            relation_strength = 0.3
            X[i, -5:] += relation_strength * X[i, :5]
            
        # 摂動とレギュラリティ
        if np.random.rand() > 0.7:
            # 特定の位置に特徴的なマーカー
            pos = np.random.randint(0, seq_length)
            X[i, pos, :] += np.random.randn(n_features) * 0.5
    
    return X, y

def train_neocortex_biokan():
    """
    NeoCortexBioKANモデルのトレーニングと評価
    """
    # データ生成
    print("複雑なシーケンスデータを生成中...")
    X_train, y_train = generate_complex_sequence_data(n_samples=800, seq_length=64, n_features=16)
    X_test, y_test = generate_complex_sequence_data(n_samples=200, seq_length=64, n_features=16)
    
    # PyTorchテンソルに変換
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # データセットとデータローダー
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # GPUが利用可能なら使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # NeoCortexBioKANモデルのインスタンス化
    in_features = X_train.shape[2]  # 特徴量の次元
    hidden_dim = 128  # 隠れ層の次元
    num_classes = 10  # クラス数
    
    model = NeoCortexBioKAN(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_blocks=3,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        max_seq_length=64,
        use_neuroplasticity=True,
        use_glia=True,
        use_working_memory=True,
        use_predictive_coding=True,
        oscillatory_dynamics=True
    )
    
    model = model.to(device)
    
    # 最適化アルゴリズムと損失関数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # トレーニングループ
    num_epochs = 10
    memory_state = None
    training_history = []
    
    print("トレーニング開始...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"エポック {epoch+1}/{num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # モデルの実行（ワーキングメモリ状態も更新）
            outputs, memory_state, pred_error, _ = model(inputs, memory_state=memory_state)
            
            # 主損失
            loss = criterion(outputs, targets)
            
            # 予測誤差も損失に加算（自由エネルギー原理の実装）
            if pred_error is not None:
                prediction_loss = 0.1 * torch.mean(torch.square(pred_error))
                loss += prediction_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 精度の計算
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
        # 精度と損失の計算
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # 履歴に保存
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy
        })
        
        print(f"エポック {epoch+1}/{num_epochs}, 損失: {avg_loss:.4f}, 精度: {accuracy:.2f}%")
        
        # エポックごとにテスト
        if (epoch + 1) % 2 == 0:
            test_neocortex_biokan(model, test_loader, device)
    
    # トレーニング履歴のプロット
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([h['epoch'] for h in training_history], [h['loss'] for h in training_history])
    plt.title('トレーニング損失')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    
    plt.subplot(1, 2, 2)
    plt.plot([h['epoch'] for h in training_history], [h['accuracy'] for h in training_history])
    plt.title('トレーニング精度')
    plt.xlabel('エポック')
    plt.ylabel('精度 (%)')
    
    plt.tight_layout()
    plt.savefig('neocortex_biokan_training_history.png')
    plt.close()
    
    # モデルの保存
    torch.save(model.state_dict(), 'neocortex_biokan_model.pth')
    print("モデルを neocortex_biokan_model.pth に保存しました")
    
    return model

def test_neocortex_biokan(model, test_loader, device):
    """
    NeoCortexBioKANモデルの評価
    """
    model.eval()
    correct = 0
    total = 0
    memory_state = None
    
    print("テスト中...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # モデルの実行
            outputs, memory_state, _, _ = model(inputs, memory_state=memory_state)
            
            # 精度の計算
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100.0 * correct / total
    print(f"テスト精度: {accuracy:.2f}%")
    
    return accuracy

def analyze_brain_simulation(model, test_loader, device):
    """
    脳シミュレーションの詳細な分析と可視化
    """
    model.eval()
    sample_inputs = None
    
    # サンプル入力の取得
    for inputs, _ in test_loader:
        sample_inputs = inputs[:1].to(device)  # 1サンプルのみ
        break
    
    if sample_inputs is None:
        print("サンプルデータが取得できませんでした")
        return
    
    print("脳活動の可視化中...")
    
    # 1. 脳波リズムの可視化
    with torch.no_grad():
        # モデルを順伝播して脳波を生成
        memory_state = None
        outputs, memory_state, pred_error, attention_maps = model(sample_inputs, memory_state=memory_state)
        
        # 各脳波タイプの活動を可視化
        plt.figure(figsize=(15, 10))
        
        wave_types = ['theta', 'alpha', 'beta', 'gamma', 'delta']
        for i, wave_type in enumerate(wave_types):
            plt.subplot(len(wave_types), 1, i+1)
            
            # 脳波ジェネレータから出力を取得
            wave_gen = model.oscillation_generators[wave_type]
            cortical_output = model.cortical_layers[-1](memory_state.mean(dim=1))
            wave = wave_gen(cortical_output).detach().cpu().numpy()[0]
            
            # 波形の生成
            seq_length = sample_inputs.size(1)
            if wave_type == 'theta':
                freq = 6
            elif wave_type == 'alpha':
                freq = 10
            elif wave_type == 'beta':
                freq = 20
            elif wave_type == 'gamma':
                freq = 40
            else:  # delta
                freq = 3
                
            t = np.linspace(0, 1, seq_length)
            wave_pattern = np.sin(2 * np.pi * freq * t)
            
            plt.plot(t, wave_pattern[:seq_length], label=f'{wave_type}波 (理論値)')
            plt.plot(t, wave.mean(axis=-1)[:seq_length], '--', label=f'{wave_type}波 (モデル)')
            plt.title(f'{wave_type}波形パターン')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig('neocortex_biokan_brain_waves.png')
        plt.close()
        
        # 2. 皮質層の活性化の可視化
        plt.figure(figsize=(15, 10))
        for i in range(6):  # 6つの皮質層
            plt.subplot(2, 3, i+1)
            
            # 各層の活性化を取得
            layer = model.cortical_layers[i]
            if i == 0:
                layer_input = memory_state.mean(dim=1)
            else:
                prev_layer = model.cortical_layers[i-1]
                layer_input = prev_layer(memory_state.mean(dim=1))
                
            layer_output = layer(layer_input).detach().cpu().numpy()[0]
            
            # ヒートマップとして表示
            plt.imshow(layer_output.reshape(8, -1), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'皮質層 {i+1} (層 {["I", "II", "III", "IV", "V", "VI"][i]})')
            
        plt.tight_layout()
        plt.savefig('neocortex_biokan_cortical_layers.png')
        plt.close()
        
        # 3. ワーキングメモリ状態の可視化
        plt.figure(figsize=(12, 6))
        
        # ワーキングメモリスロットの状態
        memory_data = memory_state.detach().cpu().numpy()[0]
        plt.subplot(1, 2, 1)
        plt.imshow(memory_data, aspect='auto', cmap='hot')
        plt.colorbar()
        plt.title('ワーキングメモリスロット')
        plt.xlabel('メモリ次元')
        plt.ylabel('スロットインデックス')
        
        # メモリ活性化の時間変化をシミュレート
        plt.subplot(1, 2, 2)
        
        # 時間経過によるメモリの減衰をシミュレート
        decay_factor = model.working_memory.decay_factor
        time_steps = 10
        memory_decay = np.zeros((time_steps, memory_data.shape[0]))
        
        current_mem = memory_data.mean(axis=1)
        for t in range(time_steps):
            memory_decay[t] = current_mem
            current_mem = current_mem * decay_factor
            
        plt.imshow(memory_decay.T, aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.title('メモリ減衰シミュレーション')
        plt.xlabel('時間ステップ')
        plt.ylabel('メモリスロット')
        
        plt.tight_layout()
        plt.savefig('neocortex_biokan_working_memory.png')
        plt.close()
        
        # 4. 高次認知機能の可視化
        plt.figure(figsize=(15, 8))
        
        # サンプル入力で高次認知機能の出力を取得
        x = model.default_mode_network(memory_state.mean(dim=1))
        
        # 各認知機能の出力
        cognition_modules = [
            ('抽象思考', model.higher_cognition.abstraction),
            ('アナロジー推論', model.higher_cognition.analogy),
            ('因果推論', model.higher_cognition.causal_reasoning),
            ('メタ認知', model.higher_cognition.metacognition),
            ('創造性', model.higher_cognition.creativity)
        ]
        
        for i, (name, module) in enumerate(cognition_modules):
            plt.subplot(2, 3, i+1)
            
            cognition_output = module(x).detach().cpu().numpy()[0]
            
            # 出力の特徴空間を2Dで視覚化
            from sklearn.decomposition import PCA
            if cognition_output.shape[0] > 2:
                pca = PCA(n_components=2)
                cognition_2d = pca.fit_transform(cognition_output.reshape(1, -1))
            else:
                cognition_2d = cognition_output[:2]
                
            # 特徴の活性化を色付きの散布図で表示
            plt.scatter(
                np.arange(len(cognition_output)), 
                cognition_output,
                c=cognition_output, 
                cmap='coolwarm', 
                alpha=0.7
            )
            plt.colorbar()
            plt.title(f'{name}の活性化パターン')
            plt.xlabel('特徴インデックス')
            plt.ylabel('活性化強度')
            
        plt.tight_layout()
        plt.savefig('neocortex_biokan_higher_cognition.png')
        plt.close()
        
        print("可視化が完了しました - PNG ファイルを確認してください")

if __name__ == "__main__":
    try:
        # モデルをトレーニング
        model = train_neocortex_biokan()
        
        # データローダーを再作成してテスト
        _, y_test = generate_complex_sequence_data(n_samples=200, seq_length=64, n_features=16)
        X_test, y_test = generate_complex_sequence_data(n_samples=200, seq_length=64, n_features=16)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # GPUが利用可能なら使用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 脳シミュレーションの分析
        analyze_brain_simulation(model, test_loader, device)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}") 