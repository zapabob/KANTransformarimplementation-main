"""
BioKANモデルの訓練と神経伝達物質の動的最適化を行うスクリプト
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from torch.amp import autocast, GradScaler
import argparse  # コマンドライン引数の解析用
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm  # 進捗バー表示用
import optuna

# 共通CUDA情報管理モジュール
from cuda_info_manager import print_cuda_info, get_device, setup_japanese_fonts

# 日本語フォントの設定（詳細表示しない）
setup_japanese_fonts(verbose=False)

# 必要なディレクトリを作成
os.makedirs('data', exist_ok=True)
os.makedirs('biokan_trained_models', exist_ok=True)
os.makedirs('biokan_results', exist_ok=True)

# デバイスの設定
device = get_device()

# ===============================================
# 生物学的注意機構 (Biological Attention Mechanism)
# ===============================================
class BiologicalAttention(nn.Module):
    """
    生物学的に着想を得た注意機構
    - 選択的注意: タスク関連情報に集中
    - 持続的注意: 長時間のタスク実行中の集中力を維持
    - 分割的注意: 複数のタスク間で注意を分割
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(BiologicalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 多頭注意機構の線形投影
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # 出力投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 局所的抑制（ラテラル抑制）- 生物学的要素
        self.lateral_inhibition = nn.Conv1d(
            in_channels=num_heads,
            out_channels=num_heads,
            kernel_size=3,
            padding=1,
            groups=num_heads
        )
        
        # 神経調節用のゲート機構
        self.neuromodulation_gate = nn.Linear(hidden_dim, num_heads)
        
        # アテンション履歴（持続的注意のため）
        self.attention_history = None
        self.history_weight = 0.3  # 履歴の重み
        
        # ベースラインアテンション（タスク無関係時）
        self.baseline = nn.Parameter(torch.ones(1, num_heads, 1) * 0.1)
    
    def forward(self, x, neuromodulation=None):
        """
        順伝播処理
        
        Args:
            x: 入力テンソル
            neuromodulation: 神経調節係数の辞書（オプション）
            
        Returns:
            注意機構を適用したテンソル
        """
        # デバッグ情報 - 静かモードで無効化
        verbose = False
        if verbose:
            print(f"BiologicalAttention input shape: {x.shape}, type: {type(x)}")
            
        batch_size, seq_len, hidden_dim = x.shape
        
        # 入力データの形状を確認
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        elif x.dim() == 3:
            batch_size = x.size(0)
        else:
            raise ValueError(f"入力データの次元が不正です: {x.shape}")
        
        # 線形投影
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # スケーリングドット積アテンション
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 神経調節効果の適用（オプション）
        if neuromodulation is not None and isinstance(neuromodulation, dict):
            # 注意調節係数を計算
            mod_input = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for nt, level in neuromodulation.items():
                if isinstance(level, (int, float)):
                    mod_input[:, :] = level
            
            mod_weights = torch.sigmoid(self.neuromodulation_gate(mod_input))
            mod_weights = mod_weights.view(batch_size, self.num_heads, 1, 1)
            
            # ノルアドレナリン効果：選択性の向上
            if 'noradrenaline' in neuromodulation:
                noradrenaline = float(neuromodulation['noradrenaline'])
                selectivity = 1.0 + 0.5 * noradrenaline
                scores = scores * selectivity
            
            # ドーパミン効果：重要情報の強調
            if 'dopamine' in neuromodulation:
                dopamine = float(neuromodulation['dopamine'])
                # トップK注意要素を強調
                top_k = max(1, int(scores.size(-1) * 0.2))  # 上位20%
                top_scores, _ = torch.topk(scores, top_k, dim=-1)
                threshold = top_scores[:, :, :, -1].unsqueeze(-1)
                emphasis = torch.where(scores >= threshold, 
                                     1.0 + 0.3 * dopamine, 
                                     torch.ones_like(scores))
                scores = scores * emphasis
        
        # ラテラル抑制の適用（生物学的側面）
        scores_pooled = scores.mean(dim=-2)  # [batch, heads, seq_len]
        lateral_inhibition = self.lateral_inhibition(scores_pooled)
        scores = scores * lateral_inhibition.unsqueeze(-2)
        
        # 持続的注意（履歴の組み込み）
        if self.attention_history is not None:
            # 前回のアテンション履歴を現在のスコアに組み込む
            history_influence = self.attention_history.detach().expand_as(scores)
            scores = (1 - self.history_weight) * scores + self.history_weight * history_influence
        
        # ソフトマックスでアテンション確率に変換
        attention_probs = F.softmax(scores, dim=-1)
        
        # この時点でのアテンション履歴を更新
        self.attention_history = attention_probs.detach().mean(dim=0, keepdim=True)
        
        # ドロップアウト
        attention_probs = self.dropout(attention_probs)
        
        # 値を加重和
        context = torch.matmul(attention_probs, v)
        
        # 次元の並べ替えと整形
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 出力投影
        output = self.output_proj(context)
        
        return output

# ===============================================
# 神経伝達物質システム (Neurotransmitter System)
# ===============================================
class DynamicNeuromodulatorSystem:
    """
    状況に応じて動的に調整される神経伝達物質システム
    """
    def __init__(self):
        # 神経伝達物質の初期レベル
        self.levels = {
            'dopamine': 0.2,     # 報酬予測、動機付け
            'serotonin': 0.3,    # 感情調整、衝動制御
            'noradrenaline': 0.4,  # 覚醒度、注意力
            'acetylcholine': 0.5,  # 記憶、学習
            'glutamate': 0.6,    # 興奮性信号
            'gaba': -0.3         # 抑制性信号
        }
        
        # 最小値と最大値の範囲
        self.ranges = {
            'dopamine': (-0.2, 1.0),
            'serotonin': (-0.5, 0.8),
            'noradrenaline': (-0.1, 0.9),
            'acetylcholine': (0.0, 0.8),
            'glutamate': (0.0, 1.0),
            'gaba': (-1.0, 0.0)
        }
        
        # 減衰率（ホメオスタシス）
        self.decay_rates = {
            'dopamine': 0.05,
            'serotonin': 0.02,
            'noradrenaline': 0.08,
            'acetylcholine': 0.04,
            'glutamate': 0.1,
            'gaba': 0.07
        }
        
        # 各神経伝達物質の履歴
        self.history = {nt: [] for nt in self.levels.keys()}
        
        # 相互作用行列：行が「元」、列が「影響先」
        self.interaction_matrix = {
            'dopamine': {'serotonin': -0.2, 'noradrenaline': 0.3},
            'serotonin': {'dopamine': -0.1, 'gaba': 0.2},
            'noradrenaline': {'acetylcholine': 0.4, 'glutamate': 0.3},
            'acetylcholine': {'glutamate': 0.2},
            'glutamate': {'gaba': -0.3},
            'gaba': {'glutamate': -0.4}
        }
    
    def update(self, context=None):
        """
        神経伝達物質レベルの更新
        
        Args:
            context: 更新の文脈情報（報酬、エラー率など）
        """
        # 現在の状態をコピー
        current_levels = self.levels.copy()
        
        # 文脈に基づく更新（与えられている場合）
        if context is not None:
            # 報酬シグナルに基づくドーパミンの更新
            if 'reward' in context:
                reward = context['reward']
                expected_reward = context.get('expected_reward', 0.0)
                # 報酬予測誤差の計算
                reward_prediction_error = reward - expected_reward
                self.levels['dopamine'] += 0.1 * reward_prediction_error
            
            # 誤差率に基づくノルアドレナリンの更新
            if 'error_rate' in context:
                error_rate = context['error_rate']
                # 高いエラー率→高い覚醒度
                self.levels['noradrenaline'] += 0.05 * error_rate
            
            # 損失変化に基づくセロトニンの更新
            if 'loss_change' in context:
                loss_change = context['loss_change']
                # 損失減少→セロトニン増加（満足度）
                if loss_change < 0:
                    self.levels['serotonin'] += 0.02 * abs(loss_change)
                else:
                    # 損失増加→セロトニン減少（不満）
                    self.levels['serotonin'] -= 0.01 * loss_change
            
            # 学習率に基づくアセチルコリンの更新
            if 'learning_phase' in context:
                learning_phase = context['learning_phase']
                # 学習初期→高いアセチルコリン（高い可塑性）
                if learning_phase < 0.3:  # 学習初期
                    self.levels['acetylcholine'] += 0.03
                elif learning_phase > 0.7:  # 学習後期
                    self.levels['acetylcholine'] -= 0.02
        
        # 神経伝達物質間の相互作用を適用
        for source, targets in self.interaction_matrix.items():
            for target, effect in targets.items():
                self.levels[target] += current_levels[source] * effect * 0.1
        
        # 自然減衰（ホメオスタシス）
        for nt in self.levels:
            decay = self.decay_rates[nt]
            self.levels[nt] -= decay * (self.levels[nt] - 0.0)  # 0.0がベースライン
        
        # 範囲内に収める
        for nt in self.levels:
            min_val, max_val = self.ranges[nt]
            self.levels[nt] = max(min_val, min(max_val, self.levels[nt]))
        
        # 履歴の更新
        for nt, level in self.levels.items():
            self.history[nt].append(level)
            # 履歴が長すぎる場合は古いものを削除
            if len(self.history[nt]) > 1000:
                self.history[nt] = self.history[nt][-1000:]
        
        return self.levels
    
    def get_levels(self):
        """現在の神経伝達物質レベルを取得"""
        return self.levels.copy()
    
    def get_history(self):
        """神経伝達物質の履歴を取得"""
        return self.history.copy()
    
    def visualize_history(self, save_path=None):
        """神経伝達物質レベルの履歴を可視化"""
        plt.figure(figsize=(10, 6))
        
        for name, values in self.history.items():
            plt.plot(values, label=name)
            
        plt.title("Neurotransmitter Levels Over Time / 神経伝達物質レベルの推移")
        plt.xlabel("Time Step / ステップ")
        plt.ylabel("Level / レベル")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()

# ===============================================
# BioKANモデル (生物学的コルモゴロフ-アーノルドネットワーク)
# ===============================================
class EnhancedBioKANModel(nn.Module):
    """
    改良版BioKANモデル
    - 動的な神経伝達物質システム
    - 生物学的注意機構
    - トレーニング中の適応学習
    """
    def __init__(self, in_features=784, hidden_dim=128, num_classes=10, num_blocks=3):
        super(EnhancedBioKANModel, self).__init__()
        self.flatten = nn.Flatten()
        
        # モデル構造パラメータ
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        
        # 入力層
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # 注意機構（生物学的）
        self.attention_layers = nn.ModuleList([
            BiologicalAttention(hidden_dim=hidden_dim, num_heads=4) 
            for _ in range(num_blocks)
        ])
        
        # BioKANブロック（処理層）
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.blocks.append(block)
        
        # 出力層
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        
        # 動的な神経伝達物質システム
        self.neuromodulator_system = DynamicNeuromodulatorSystem()
        
        # 損失追跡
        self.last_loss = None
        
        # 学習フェーズトラッカー（0〜1）
        self.learning_phase = 0.0
        
        # 統計追跡
        self.stats = {
            'training_errors': [],
            'validation_errors': [],
            'rewards': []
        }
    
    def forward(self, x, neuromodulators=None, **kwargs):
        """
        順伝播処理
        
        Args:
            x: 入力テンソル
            neuromodulators: 神経調節係数の辞書（オプション）
            **kwargs: その他のパラメータ
            
        Returns:
            モデルの出力
        """
        # デバッグ情報 - 静かモードで無効化
        verbose = False
        if verbose:
            print(f"EnhancedBioKANModel input shape: {x.shape}, type: {type(x)}")
            
        # 入力の形状を確認
        is_3d = False
        if x.dim() == 3:
            is_3d = True
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        # 入力投影
        x = F.relu(self.input_proj(x))
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 層の活性化を追跡（必要な場合）
        activations = {'input': x.detach()} if kwargs.get('return_activations', False) else None
        
        # 現在の神経伝達物質レベル
        nt_levels = self.neuromodulator_system.get_levels()
        
        # 各BioKANブロックを通過
        for i, (block, attention) in enumerate(zip(self.blocks, self.attention_layers)):
            # 生物学的注意機構の適用
            attended_x = attention(x, nt_levels)
            
            # 神経調節の影響をシミュレート
            attention_factor = 1.0 + 0.2 * nt_levels['noradrenaline']  # ノルアドレナリン：注意調整
            learning_factor = 1.0 + 0.3 * nt_levels['acetylcholine']   # アセチルコリン：学習促進
            inhibition_factor = 1.0 + 0.5 * nt_levels['gaba']         # GABA：活動抑制
            excitation_factor = 1.0 + 0.4 * nt_levels['glutamate']    # グルタミン酸：活動増強
            
            # ブロック出力に調節効果を適用
            block_output = block(attended_x)
            
            # 興奮性/抑制性バランスの調整
            x = x + (block_output * excitation_factor * learning_factor + 
                    attended_x * attention_factor * inhibition_factor)
            
            # 活性化の追跡（要求時）
            if kwargs.get('return_activations', False):
                activations[f'block_{i}'] = x.detach()
        
        # 出力投影のための形状変更
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        output = self.output_proj(x)
        
        if kwargs.get('return_activations', False):
            activations['output'] = output.detach()
            return output, activations
        
        return output
    
    def update_neuromodulators(self, context):
        """学習コンテキストに基づいて神経伝達物質を更新"""
        return self.neuromodulator_system.update(context)
    
    def get_neuromodulator_levels(self):
        """現在の神経伝達物質レベルを取得"""
        return self.neuromodulator_system.get_levels()
    
    def set_learning_phase(self, current_epoch, total_epochs):
        """学習フェーズの更新（0〜1）"""
        self.learning_phase = current_epoch / total_epochs

# ===============================================
# 評価指標（決定係数を含む）
# ===============================================
def calculate_r_squared(y_true, y_pred):
    """
    決定係数（R²）を計算する。
    神経科学研究では、モデルがデータの分散をどれだけ説明できるかを示す指標としてよく使用される。
    
    Args:
        y_true: 真の値（one-hotエンコーディングまたはインデックス）
        y_pred: 予測値（確率または生の出力）
        
    Returns:
        決定係数（R²）
    """
    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
        # ラベルがインデックス形式の場合、one-hotエンコーディングに変換
        y_true_onehot = torch.zeros(y_true.size(0), y_pred.size(1), device=y_true.device)
        y_true_onehot.scatter_(1, y_true.unsqueeze(1), 1)
    else:
        y_true_onehot = y_true
    
    # 確率分布に変換（必要な場合）
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        # softmaxがまだ適用されていない場合は適用
        if torch.max(y_pred).item() > 10:  # logits判定の簡易方法
            y_pred = F.softmax(y_pred, dim=1)
    
    # 平均値の計算
    y_mean = torch.mean(y_true_onehot, dim=0)
    
    # 全平方和 (Total Sum of Squares)
    SS_tot = torch.sum((y_true_onehot - y_mean) ** 2)
    
    # 残差平方和 (Residual Sum of Squares)
    SS_res = torch.sum((y_true_onehot - y_pred) ** 2)
    
    # 決定係数（R²）の計算
    r_squared = 1 - (SS_res / SS_tot)
    
    # NaNを0で置換
    r_squared = torch.nan_to_num(r_squared, nan=0.0)
    
    # 平均R²を返す
    return r_squared.mean().item()

# ===============================================
# 訓練ループ関数
# ===============================================
def train_enhanced_biokan(model, train_loader, val_loader, device, 
                        epochs=20, lr=0.001, weight_decay=1e-5,
                        save_dir='biokan_trained_models', gpu_cache_clear=False):
    """
    拡張BioKANモデルのトレーニング関数
    神経伝達物質の動的調整とその効果を含む
    """
    # 訓練開始時間
    start_time = time.time()
    
    # GPUメモリ統計
    gpu_stats = {
        'memory_usage': [],
        'time_per_epoch': []
    }
    
    # 損失関数とオプティマイザー
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 混合精度トレーニング用のスケーラー
    scaler = GradScaler()
    
    # 訓練履歴
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_r2': [],  # 訓練データの決定係数
        'val_r2': [],    # 検証データの決定係数
        'neurotransmitter_levels': []
    }
    
    # 最良モデルの追跡
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, 'best_biokan_model.pth')
    
    # CUDA設定の最適化
    if device.type == 'cuda':
        # 現在のGPUデバイス情報を表示
        current_device = torch.cuda.current_device()
        print(f"現在使用中のGPU: {torch.cuda.get_device_name(current_device)}")
        print(f"GPU メモリ使用量: {torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB")
        print(f"GPU キャッシュ: {torch.cuda.memory_reserved(current_device) / 1024**2:.2f} MB")
        
        # CUDA 12向けの追加最適化
        if torch.version.cuda.startswith('12.'):
            print("CUDA 12向け追加最適化を適用します")
            # JIT コンパイルの有効化（一部の操作を高速化）
            torch.jit.enable_onednn_fusion(True)
            # 非同期CUDA操作の有効化
            torch.cuda.set_sync_debug_mode(0)
        
        # キャッシュのクリア
        torch.cuda.empty_cache()
        
        # カーネル実行を非同期化
        torch.backends.cudnn.benchmark = True
    
    # エポック毎の訓練
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # GPU使用量を記録
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated(device) / 1024**2
            gpu_stats['memory_usage'].append(current_memory)
            print(f"エポック開始時GPU使用量: {current_memory:.2f} MB")
        
        # 学習フェーズの更新
        model.set_learning_phase(epoch, epochs)
        
        # ===== 訓練フェーズ =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_outputs_all = []
        train_targets_all = []
        
        # tqdmを使用したプログレスバー
        train_pbar = tqdm(train_loader, desc=f'エポック {epoch+1}/{epochs} [訓練]', 
                         leave=True, dynamic_ncols=True)
        
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.shape[0]
            
            # 勾配をゼロにリセット
            optimizer.zero_grad()
            
            # 混合精度を使用した順伝播
            with autocast(enabled=device.type == 'cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 決定係数の計算のために保存
            train_outputs_all.append(outputs.detach())
            train_targets_all.append(targets)
            
            # スケーラーを使用した逆伝播と最適化
            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # 統計更新
            train_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # バッチごとの誤差率
            batch_error_rate = 1.0 - (predicted.eq(targets).sum().item() / targets.size(0))
            
            # 損失の変化を計算（最初のバッチでは基準値を設定）
            if model.last_loss is None:
                loss_change = 0.0
            else:
                loss_change = loss.item() - model.last_loss
            model.last_loss = loss.item()
            
            # 神経伝達物質の更新コンテキスト
            neuromodulator_context = {
                'error_rate': batch_error_rate,
                'loss_change': loss_change,
                'learning_phase': model.learning_phase,
                'reward': 1.0 - batch_error_rate  # 精度を報酬として使用
            }
            
            # 神経伝達物質レベルの更新
            nt_levels = model.update_neuromodulators(neuromodulator_context)
            
            # tqdmプログレスバーの更新
            curr_acc = train_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{curr_acc:.4f}',
                'NA': f'{nt_levels["noradrenaline"]:.2f}',
                'DA': f'{nt_levels["dopamine"]:.2f}'
            })
        
        # エポック全体の訓練データについて決定係数を計算
        train_outputs_all = torch.cat(train_outputs_all, dim=0)
        train_targets_all = torch.cat(train_targets_all, dim=0)
        train_r2 = calculate_r_squared(train_targets_all, train_outputs_all)
        
        # エポック平均の訓練損失と精度
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # ===== 検証フェーズ =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_outputs_all = []
        val_targets_all = []
        
        # tqdmを使用した検証プログレスバー
        val_pbar = tqdm(val_loader, desc=f'エポック {epoch+1}/{epochs} [検証]', 
                       leave=True, dynamic_ncols=True)
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)
                
                # データの形状整形（必要な場合）
                if inputs.dim() > 2:
                    inputs = inputs.view(batch_size, -1)
                
                # 混合精度を使用した順伝播
                with autocast(enabled=device.type == 'cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # 決定係数の計算のために保存
                val_outputs_all.append(outputs)
                val_targets_all.append(targets)
                
                # 統計更新
                val_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # tqdmプログレスバーの更新
                curr_val_acc = val_correct / val_total
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{curr_val_acc:.4f}'
                })
        
        # エポック全体の検証データについて決定係数を計算
        val_outputs_all = torch.cat(val_outputs_all, dim=0)
        val_targets_all = torch.cat(val_targets_all, dim=0)
        val_r2 = calculate_r_squared(val_targets_all, val_outputs_all)
        
        # エポック平均の検証損失と精度
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 学習率スケジューラの更新
        scheduler.step(val_loss)
        
        # 履歴の更新
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['neurotransmitter_levels'].append(model.get_neuromodulator_levels())
        
        # エポック結果の表示（決定係数を目立たせる）
        print(f'\nエポック {epoch+1}/{epochs} 結果:')
        print(f'  訓練: 損失 {train_loss:.4f} | 精度 {train_acc:.4f} | R² {train_r2:.4f}')
        print(f'  検証: 損失 {val_loss:.4f} | 精度 {val_acc:.4f} | R² {val_r2:.4f}')
        
        # 決定係数の詳細な説明
        if epoch == 0 or epoch == epochs - 1:
            print(f'\n決定係数(R²)の解釈:')
            print(f'  R² = 1.0: モデルが完全に予測できている（理想的）')
            print(f'  R² > 0.7: 強い説明力を持つモデル')
            print(f'  R² > 0.5: 中程度の説明力を持つモデル')
            print(f'  R² < 0.3: 弱い説明力（改善の余地あり）')
        
        # エポック完了時間を記録
        epoch_time = time.time() - epoch_start_time
        if device.type == 'cuda':
            gpu_stats['time_per_epoch'].append(epoch_time)
            print(f"エポック実行時間: {epoch_time:.2f}秒")
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'モデルを保存しました: {best_model_path}')
        
        # オプションでGPUキャッシュをクリア
        if gpu_cache_clear and device.type == 'cuda':
            torch.cuda.empty_cache()
            print("GPUキャッシュをクリアしました")
    
    # 訓練完了時間
    total_time = time.time() - start_time
    print(f"総訓練時間: {total_time:.2f}秒 (平均: {total_time/epochs:.2f}秒/エポック)")
    
    # 最終的な決定係数の詳細な表示
    print("\n=== 訓練終了後の最終評価 ===")
    print(f"訓練データに対する決定係数(R²): {train_r2:.6f}")
    print(f"検証データに対する決定係数(R²): {val_r2:.6f}")
    r2_diff = abs(train_r2 - val_r2)
    if r2_diff > 0.1:
        print(f"警告: 訓練と検証の決定係数の差が大きい ({r2_diff:.4f}) - 過学習の可能性があります")
    
    # GPU使用統計のプロット
    if device.type == 'cuda':
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(gpu_stats['memory_usage'])
        plt.xlabel('エポック')
        plt.ylabel('メモリ使用量 (MB)')
        plt.title('GPU メモリ使用量の推移')
        
        plt.subplot(1, 2, 2)
        plt.plot(gpu_stats['time_per_epoch'])
        plt.xlabel('エポック')
        plt.ylabel('実行時間 (秒)')
        plt.title('エポックごとの実行時間')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gpu_stats.png'))
        plt.close()
    
    # 訓練終了後の神経伝達物質レベルの可視化
    model.neuromodulator_system.visualize_history(
        save_path=os.path.join(save_dir, 'neurotransmitter_history.png')
    )
    
    # 学習曲線のプロット（損失、精度、決定係数）
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='訓練損失')
    plt.plot(history['val_loss'], label='検証損失')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.legend()
    plt.title('学習曲線 - 損失')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='訓練精度')
    plt.plot(history['val_acc'], label='検証精度')
    plt.xlabel('エポック')
    plt.ylabel('精度')
    plt.legend()
    plt.title('学習曲線 - 精度')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_r2'], label='訓練R²')
    plt.plot(history['val_r2'], label='検証R²')
    plt.xlabel('エポック')
    plt.ylabel('決定係数（R²）')
    plt.legend()
    plt.title('学習曲線 - 決定係数')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()
    
    return history, best_model_path

# ===============================================
# メイン実行関数
# ===============================================
def main():
    """メイン実行関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='BioKANモデルのトレーニングスクリプト')
    parser.add_argument('--epochs', type=int, default=15, help='訓練エポック数')
    parser.add_argument('--batch_size', type=int, default=128, help='バッチサイズ')
    parser.add_argument('--test_batch_size', type=int, default=256, help='テスト時のバッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='重み減衰')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隠れ層の次元')
    parser.add_argument('--num_blocks', type=int, default=3, help='BioKANブロック数')
    parser.add_argument('--save_dir', type=str, default='biokan_trained_models', help='モデル保存ディレクトリ')
    parser.add_argument('--gpu_cache_clear', action='store_true', help='各エポック後にGPUキャッシュをクリア')
    parser.add_argument('--distributed', action='store_true', help='分散処理モードを有効化')
    parser.add_argument('--local_rank', type=int, default=0, help='分散処理用ローカルランク（自動設定）')
    args = parser.parse_args()
    
    # モデルのハイパーパラメータ
    params = {
        'in_features': 28*28,  # MNIST画像サイズ
        'hidden_dim': args.hidden_dim,
        'num_classes': 10,
        'num_blocks': args.num_blocks,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'test_batch_size': args.test_batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'save_dir': args.save_dir,
        'gpu_cache_clear': args.gpu_cache_clear,
        'distributed': args.distributed,
        'local_rank': args.local_rank
    }
    
    # 保存ディレクトリの作成
    os.makedirs(params['save_dir'], exist_ok=True)
    
    # ハイパーパラメータの保存
    with open(os.path.join(params['save_dir'], 'hyperparameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # データセットの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # 訓練データの分割（訓練と検証）
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params['batch_size'], 
        shuffle=True,
        pin_memory=device.type == 'cuda',  # GPU使用時はピンメモリを有効化
        num_workers=4 if device.type == 'cuda' else 0  # GPU使用時はワーカー数を増やす
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params['batch_size'],
        pin_memory=device.type == 'cuda',
        num_workers=4 if device.type == 'cuda' else 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=params['test_batch_size'],  # テスト時は大きなバッチサイズを使用
        pin_memory=device.type == 'cuda',
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # 拡張BioKANモデルの作成
    model = EnhancedBioKANModel(
        in_features=params['in_features'],
        hidden_dim=params['hidden_dim'],
        num_classes=params['num_classes'],
        num_blocks=params['num_blocks']
    )
    
    # 分散処理の設定
    if args.distributed and torch.cuda.device_count() > 1:
        # 分散処理初期化
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        model = model.to(device)
        # モデルをDDPでラップ
        model = DDP(model, device_ids=[args.local_rank])
        print(f"分散処理モード: {dist.get_rank()}/{dist.get_world_size()}")
    else:
        model = model.to(device)
    
    # モデル情報の表示
    print(f"拡張BioKANモデル構成:")
    print(f"  入力特徴数: {params['in_features']}")
    print(f"  隠れ層次元: {params['hidden_dim']}")
    print(f"  出力クラス数: {params['num_classes']}")
    print(f"  BioKANブロック数: {params['num_blocks']}")
    print(f"  合計パラメータ数: {sum(p.numel() for p in model.parameters())}")
    print(f"  デバイス: {device}")
    if torch.cuda.device_count() > 1:
        print(f"  使用GPU数: {torch.cuda.device_count()}")
    
    # モデルの訓練
    print("\nモデル訓練開始...")
    history, best_model_path = train_enhanced_biokan(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=params['epochs'],
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
        save_dir=params['save_dir'],
        gpu_cache_clear=params['gpu_cache_clear']
    )
    
    # 最良モデルの読み込み
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # テストセットでの評価
    print("\nテストセットでの評価...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_outputs_all = []
    test_targets_all = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # データの形状整形（必要な場合）
            if inputs.dim() > 2:
                inputs = inputs.view(inputs.size(0), -1)
            
            # 混合精度を使用した順伝播
            with autocast(enabled=device.type == 'cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 決定係数の計算のために保存
            test_outputs_all.append(outputs)
            test_targets_all.append(targets)
            
            # 統計更新
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    # テスト全体の決定係数を計算
    test_outputs_all = torch.cat(test_outputs_all, dim=0)
    test_targets_all = torch.cat(test_targets_all, dim=0)
    test_r2 = calculate_r_squared(test_targets_all, test_outputs_all)
    
    # テスト結果の計算と表示
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total
    
    print(f"テスト損失: {test_loss:.4f}")
    print(f"テスト精度: {test_acc:.4f}")
    print(f"テスト決定係数(R²): {test_r2:.4f}")
    
    # テスト結果の保存
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_r_squared': test_r2,
        'samples': len(test_loader.dataset)
    }
    
    with open(os.path.join(params['save_dir'], 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # 決定係数の追加可視化
    class_r2_values = []
    
    # クラスごとの決定係数を計算
    test_probs = F.softmax(test_outputs_all, dim=1)
    for cls in range(10):  # MNIST has 10 classes
        # クラスcの正解ラベル
        cls_targets = (test_targets_all == cls).float()
        # クラスcの予測確率
        cls_preds = test_probs[:, cls]
        # 決定係数の計算（1次元なので簡略化）
        y_mean = torch.mean(cls_targets)
        SS_tot = torch.sum((cls_targets - y_mean) ** 2)
        SS_res = torch.sum((cls_targets - cls_preds) ** 2)
        cls_r2 = 1 - (SS_res / SS_tot)
        cls_r2 = torch.nan_to_num(cls_r2, nan=0.0).item()
        class_r2_values.append(cls_r2)
    
    # クラスごとの決定係数をプロット
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), class_r2_values)
    plt.xlabel('数字クラス')
    plt.ylabel('決定係数（R²）')
    plt.title('クラスごとの決定係数（R²）')
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.3)
    
    for i, r2 in enumerate(class_r2_values):
        plt.text(i, r2 + 0.01, f'{r2:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(params['save_dir'], 'class_r_squared.png'))
    plt.close()
    
    print(f"訓練とテストが完了しました。結果は {params['save_dir']} に保存されています。")

if __name__ == "__main__":
    main() 