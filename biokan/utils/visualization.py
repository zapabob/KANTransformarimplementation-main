"""
BioKANモデルの可視化ユーティリティ
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import io
import base64
try:
    from IPython.display import HTML
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    print("Warning: IPython not found. Some visualization features will be limited.")


def visualize_layer_activations(model, sample_input, layer_indices=None, cmap='viridis'):
    """
    BioKANモデルの各層の活性化を可視化
    
    Args:
        model: BioKANモデル
        sample_input: サンプル入力
        layer_indices: 可視化する層のインデックスリスト
        cmap: カラーマップ
    
    Returns:
        fig: Matplotlib図オブジェクト
    """
    # 推論モード
    model.eval()
    
    # 順伝播実行（勾配計算なし）
    with torch.no_grad():
        # 入力変換
        h = model.input_transform(sample_input)
        
        # 層出力を記録
        layer_outputs = [h.cpu().numpy()]
        
        # 各ブロックを処理
        for block in model.blocks:
            h = block(h)
            layer_outputs.append(h.cpu().numpy())
    
    # 可視化する層のインデックスを決定
    if layer_indices is None:
        layer_indices = range(len(layer_outputs))
    
    # 最初のサンプルのみ可視化
    sample_idx = 0
    
    # 可視化
    n_layers = len(layer_indices)
    fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 4, 4))
    
    if n_layers == 1:
        axes = [axes]
    
    for i, layer_idx in enumerate(layer_indices):
        if layer_idx < len(layer_outputs):
            # 層の活性化を取得
            activation = layer_outputs[layer_idx][sample_idx]
            
            # 2Dグリッドに変形（近似的な視覚化のため）
            grid_size = int(np.ceil(np.sqrt(activation.shape[0])))
            padding = grid_size * grid_size - activation.shape[0]
            
            if padding > 0:
                activation = np.pad(activation, (0, padding), 'constant', constant_values=np.nan)
            
            activation_grid = activation.reshape(grid_size, grid_size)
            
            # ヒートマップ表示
            im = axes[i].imshow(activation_grid, cmap=cmap)
            axes[i].set_title(f'Layer {layer_idx}')
            fig.colorbar(im, ax=axes[i])
            axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_astrocyte_calcium(astrocyte, ax=None, cmap='hot', title=None):
    """
    アストロサイトのカルシウム活動をプロット
    
    Args:
        astrocyte: Astrocyteオブジェクト
        ax: Matplotlibの軸オブジェクト
        cmap: カラーマップ
        title: プロットのタイトル
    
    Returns:
        ax: 軸オブジェクト
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    # カルシウム活動を取得
    calcium = astrocyte.activation
    
    # ヒートマップとしてプロット
    im = ax.imshow(calcium, cmap=cmap, interpolation='nearest')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('アストロサイトCa²⁺活動')
    
    plt.colorbar(im, ax=ax)
    return ax


def visualize_cortical_layers(model, sample_input, save_path=None):
    """
    皮質層の活動を可視化
    
    Args:
        model: BioKANモデル
        sample_input: サンプル入力
        save_path: 結果を保存するパス（Noneの場合は表示のみ）
    
    Returns:
        fig: Matplotlib図オブジェクト
    """
    # 推論モード設定
    model.eval()
    
    # 皮質層活動を取得
    with torch.no_grad():
        # 入力変換
        h = model.input_transform(sample_input)
        
        # ブロック出力を取得
        block_outputs = []
        for block in model.blocks:
            h = block(h)
            block_outputs.append(h.detach())
        
        # ブロック出力を6つの皮質層に分割
        num_blocks = len(block_outputs)
        layer_activities = []
        
        if num_blocks >= 3:
            # 複数のブロック出力から分割
            blocks_per_layer = max(1, num_blocks // 6)
            remaining_layers = 6 - (num_blocks // blocks_per_layer)
            
            for i, block_output in enumerate(block_outputs):
                for j in range(blocks_per_layer):
                    if len(layer_activities) < 6:
                        layer_h = torch.tanh(block_output + 0.1 * j)
                        layer_activities.append(layer_h)
            
            for i in range(remaining_layers):
                if len(layer_activities) < 6:
                    layer_h = torch.sigmoid(block_outputs[-1] + 0.05 * i)
                    layer_activities.append(layer_h)
        else:
            # ブロック数が少ない場合
            for i, block_output in enumerate(block_outputs):
                num_layers = 6 // num_blocks
                for j in range(num_layers):
                    if len(layer_activities) < 6:
                        act_fn = torch.tanh if j % 2 == 0 else torch.sigmoid
                        layer_h = act_fn(block_output + 0.1 * j)
                        layer_activities.append(layer_h)
            
            while len(layer_activities) < 6:
                j = len(layer_activities) - (6 // num_blocks) * (num_blocks - 1)
                layer_h = torch.relu(block_outputs[-1] + 0.05 * j)
                layer_activities.append(layer_h)
    
    # 皮質6層の可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 各層の活動を可視化
    for i, layer_activity in enumerate(layer_activities):
        # 最初のサンプルの活動を取得
        activity = layer_activity[0].cpu().numpy()
        
        # 2Dグリッドに変形（近似的な視覚化）
        grid_size = int(np.ceil(np.sqrt(activity.shape[0])))
        padding = grid_size * grid_size - activity.shape[0]
        
        if padding > 0:
            activity = np.pad(activity, (0, padding), 'constant', constant_values=np.nan)
        
        activity_grid = activity.reshape(grid_size, grid_size)
        
        # 層の名前（神経科学的な皮質層名）
        layer_names = ['Layer I', 'Layer II/III', 'Layer IV', 'Layer V', 'Layer VI', 'Layer VIb']
        
        # ヒートマップとしてプロット
        im = axes[i].imshow(activity_grid, cmap='viridis')
        axes[i].set_title(f'{layer_names[i]}')
        fig.colorbar(im, ax=axes[i])
    
    plt.suptitle('皮質6層の活動パターン', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def animate_calcium_waves(astrocyte, n_frames=30, interval=100):
    """
    アストロサイトのカルシウム波の時間的変化をアニメーション化
    
    Args:
        astrocyte: Astrocyteオブジェクト
        n_frames: アニメーションのフレーム数
        interval: フレーム間の間隔（ミリ秒）
    
    Returns:
        HTML: アニメーションを表示するHTMLオブジェクト
    """
    # 初期状態を保存
    initial_activation = astrocyte.activation.copy()
    
    # 拡散シミュレーション用の関数
    def simulate_diffusion(activation, diffusion_rate=0.1, decay_rate=0.05):
        # パッディング
        padded = np.pad(activation, 1, mode='constant')
        
        # 拡散カーネル
        kernel = np.array([[0.05, 0.1, 0.05], 
                           [0.1, 0.4, 0.1], 
                           [0.05, 0.1, 0.05]])
        
        # 出力配列
        diffused = np.zeros_like(activation)
        
        # 2D拡散
        region_shape = activation.shape
        for i in range(region_shape[0]):
            for j in range(region_shape[1]):
                diffused[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        
        # 拡散と減衰の適用
        result = (1 - diffusion_rate) * activation + diffusion_rate * diffused
        result *= (1 - decay_rate)
        
        return result
    
    # アニメーション用の図と軸
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 初期画像
    calcium = initial_activation.copy()
    im = ax.imshow(calcium, cmap='hot', interpolation='nearest')
    ax.set_title('アストロサイトCa²⁺波伝播')
    plt.colorbar(im, ax=ax)
    
    # アニメーション更新関数
    def update(frame):
        nonlocal calcium
        
        # 新しいカルシウム波の発生（ランダムな場所）
        if frame % 10 == 0:
            i, j = np.random.randint(0, calcium.shape[0]), np.random.randint(0, calcium.shape[1])
            calcium[i, j] += 1.0
        
        # 拡散シミュレーション
        calcium = simulate_diffusion(calcium)
        
        # 更新した画像の設定
        im.set_array(calcium)
        return [im]
    
    # アニメーション作成
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    
    # HTMLとしてアニメーションを表示
    if HAS_IPYTHON:
        plt.close()
        return HTML(ani.to_html5_video())
    else:
        # IPythonがない場合、アニメーションを表示
        plt.show()
        return ani


def plot_layer_temporal_integration(model, ax=None):
    """
    皮質層間の時間的統合効果をプロット
    
    Args:
        model: BioKANモデル
        ax: Matplotlibの軸オブジェクト
    
    Returns:
        ax: 軸オブジェクト
    """
    if not hasattr(model, 'cortical_layer_connections'):
        raise ValueError("モデルに皮質層間接続が実装されていません。")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # 皮質層間の接続を取得
    connections = torch.sigmoid(model.cortical_layer_connections).detach().cpu().numpy()
    
    # ヒートマップとして表示
    layer_names = ['I', 'II/III', 'IV', 'V', 'VI', 'VIb']
    sns.heatmap(connections, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=layer_names, yticklabels=layer_names, ax=ax)
    
    ax.set_title("皮質層間時間差統合強度")
    ax.set_xlabel("受信層")
    ax.set_ylabel("送信層")
    
    return ax 