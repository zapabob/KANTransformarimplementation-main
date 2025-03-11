"""
BioKANモデルの中核実装
コルモゴロフ・アーノルド・ネットワークを拡張して生体模倣的アーキテクチャを提供
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from biokan.core.layers import TernaryActivationFunction, MultiHeadAttention, KANLinear
from biokan.neuro.neuromodulators import NeuromodulatorSystem
from biokan.neuro.glial_cells import Astrocyte, Microglia


class BiologicalMultiHeadAttention(nn.Module):
    """
    生物学的な特性を持つマルチヘッドアテンション
    アストロサイトの調節効果と神経伝達物質の影響を取り入れる
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_neuromodulation=True):
        """生物学的マルチヘッドアテンションの初期化

        Args:
            embed_dim (int): 埋め込み次元
            num_heads (int): アテンションヘッドの数
            dropout (float, optional): ドロップアウト率。デフォルトは0.0。
            use_neuromodulation (bool, optional): 神経調節を使用するかどうか。デフォルトはTrue。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_neuromodulation = use_neuromodulation
        
        # 注意機構の重みを初期化
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # 神経調節用のパラメータ
        if use_neuromodulation:
            self.neuromodulation = NeuromodulationSystem(embed_dim)
            self.attention_scale = nn.Parameter(torch.ones(1))
            self.attention_bias = nn.Parameter(torch.zeros(1))

    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size = query.size(0)
        seq_len = query.size(1) if len(query.size()) > 2 else 1
        
        # シーケンス長が1の場合、次元を追加
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        
        # 線形変換
        q = self.q_proj(query)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(key)    # [batch_size, seq_len, embed_dim]
        v = self.v_proj(value)  # [batch_size, seq_len, embed_dim]
        
        # ヘッドごとに分割
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # スケーリングされた内積注意
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 値との積
        attn = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 神経調節の適用
        if self.use_neuromodulation:
            neuromod = self.neuromodulation(query.view(batch_size * seq_len, -1))  # [batch_size * seq_len, embed_dim]
            neuromod = neuromod.view(batch_size, seq_len, -1)  # [batch_size, seq_len, embed_dim]
            neuromod = neuromod.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            attn = attn * self.attention_scale + self.attention_bias
            attn = attn * (1 + neuromod)
        
        # 形状の復元
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 出力の線形変換
        output = self.out_proj(attn)
        
        # シーケンス長が1の場合、次元を削除
        if seq_len == 1:
            output = output.squeeze(1)
        
        if need_weights:
            return output, attn_weights
        return output


class BioKANBlock(nn.Module):
    """BioKANの基本構成ブロック"""
    
    def __init__(
        self,
        hidden_dim,
        num_heads=4,
        dropout=0.1,
        attention_type="biological",
        device="cuda",
        use_neuromodulation=True,
        use_batch_norm=True
    ):
        """BioKANブロックの初期化

        Args:
            hidden_dim (int): 隠れ層の次元数
            num_heads (int, optional): アテンションヘッドの数。デフォルトは4。
            dropout (float, optional): ドロップアウト率。デフォルトは0.1。
            attention_type (str, optional): アテンションのタイプ。デフォルトは"biological"。
            device (str, optional): 使用するデバイス。デフォルトは"cuda"。
            use_neuromodulation (bool, optional): 神経調節を使用するかどうか。デフォルトはTrue。
            use_batch_norm (bool, optional): バッチ正規化を使用するかどうか。デフォルトはTrue。
        """
        super().__init__()
        
        self.device = device
        self.use_batch_norm = use_batch_norm
        
        # hidden_dimをnum_headsで割り切れるように調整
        self.hidden_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads
        
        # 特徴アテンション
        self.feature_attention = BiologicalMultiHeadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_neuromodulation=use_neuromodulation
        )
        
        # 入力変換（必要な場合）
        if hidden_dim != self.hidden_dim:
            self.input_projection = nn.Linear(hidden_dim, self.hidden_dim)
        else:
            self.input_projection = nn.Identity()
        
        # 出力変換（必要な場合）
        if hidden_dim != self.hidden_dim:
            self.output_projection = nn.Linear(self.hidden_dim, hidden_dim)
        else:
            self.output_projection = nn.Identity()
        
        # バッチ正規化
        if use_batch_norm:
            self.input_norm = nn.BatchNorm1d(self.hidden_dim)
            self.attention_norm = nn.BatchNorm1d(self.hidden_dim)
        
        # その他のレイヤー
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 神経調節システム
        self.use_neuromodulation = use_neuromodulation
        if use_neuromodulation:
            self.neuromodulation = NeuromodulationSystem(self.hidden_dim, device=device)
        
        # デバイスに移動
        self.to(device)
    
    def forward(self, x):
        """順伝播処理

        Args:
            x (torch.Tensor): 入力テンソル [batch_size x hidden_dim]

        Returns:
            torch.Tensor: 出力テンソル [batch_size x hidden_dim]
        """
        # 入力の変換（必要な場合）
        h = self.input_projection(x)
        
        # バッチ正規化（使用する場合）
        if self.use_batch_norm:
            if len(h.shape) == 2:
                h = self.input_norm(h)
            else:
                h = h.transpose(1, 2)
                h = self.input_norm(h)
                h = h.transpose(1, 2)
        
        # レイヤー正規化
        h = self.layer_norm(h)
        
        # セルフアテンションの適用
        h = self.feature_attention(h)
        
        # バッチ正規化（使用する場合）
        if self.use_batch_norm:
            if len(h.shape) == 2:
                h = self.attention_norm(h)
            else:
                h = h.transpose(1, 2)
                h = self.attention_norm(h)
                h = h.transpose(1, 2)
        
        # ドロップアウト
        h = self.dropout(h)
        
        # 出力の変換（必要な場合）
        h = self.output_projection(h)
        
        # 残差接続
        h = h + x
        
        return h


class CorticalAttention(nn.Module):
    """皮質型アテンションメカニズム"""
    
    def __init__(self, embed_dim, num_regions=4, region_heads=2, dropout=0.1):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_regions: 脳領域の数（前頭前皮質、頭頂葉、側頭葉など）
            region_heads: 各領域のアテンションヘッド数
            dropout: ドロップアウト確率
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_regions = num_regions
        self.region_heads = region_heads
        
        # 各「脳領域」のアテンション
        self.region_attentions = nn.ModuleList([
            BiologicalMultiHeadAttention(
                embed_dim // num_regions,
                region_heads,
                dropout=dropout
            )
            for _ in range(num_regions)
        ])
        
        # 領域間統合のためのアテンション
        self.integration_attention = BiologicalMultiHeadAttention(
            embed_dim, num_heads=num_regions, dropout=dropout
        )
        
        # 出力投影
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, embed_dim]
            
        Returns:
            出力テンソル [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 入力を「脳領域」に分割
        region_size = self.embed_dim // self.num_regions
        regions = x.view(batch_size, seq_len, self.num_regions, region_size)
        
        # 各領域で独立にアテンション計算
        region_outputs = []
        for i, attention in enumerate(self.region_attentions):
            region_input = regions[:, :, i, :]  # [batch_size, seq_len, region_size]
            
            # 各領域内でのセルフアテンション
            region_output, _ = attention(region_input, region_input, region_input)
            region_outputs.append(region_output)
        
        # 領域出力を連結
        concatenated = torch.cat(region_outputs, dim=-1)  # [batch_size, seq_len, embed_dim]
        
        # 領域間の情報統合（階層的注意）
        integrated, _ = self.integration_attention(concatenated, concatenated, concatenated)
        
        # 最終出力投影
        output = self.output_projection(integrated)
        output = self.dropout(output)
        
        return output


class HierarchicalMultiScaleAttention(nn.Module):
    """階層的マルチスケールアテンション"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_scales=3):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_heads: ヘッド数
            dropout: ドロップアウト確率
            num_scales: スケール数（時間・空間スケール）
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        
        # マルチスケールアテンション
        # 各スケールは異なる範囲の情報に焦点を当てる
        self.scale_attentions = nn.ModuleList([
            BiologicalMultiHeadAttention(
                embed_dim,
                num_heads,
                dropout=dropout
            )
            for _ in range(num_scales)
        ])
        
        # スケール重み（学習可能）
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 出力投影
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, embed_dim]
            mask: アテンションマスク
            
        Returns:
            出力テンソル [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 各スケールでのアテンション計算
        scale_outputs = []
        
        for i, attention in enumerate(self.scale_attentions):
            # スケールに応じたアテンションマスクの調整
            scale_mask = mask
            
            if mask is not None and i > 0:
                # 大きなスケールでは、より広い範囲に注目できるようにマスクを調整
                # （実装例：スケールに応じて異なる数のトークンを参照可能にする）
                pass
            
            # アテンション計算
            scale_output, _ = attention(x, x, x, attn_mask=scale_mask)
            scale_outputs.append(scale_output)
        
        # スケール重みのソフトマックス正規化
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        # 重み付き集約
        output = torch.zeros_like(x)
        for i, scale_output in enumerate(scale_outputs):
            output += scale_weights[i] * scale_output
        
        # 最終出力投影
        output = self.output_projection(output)
        output = self.dropout(output)
        
        return output


class EnhancedAstrocyte(nn.Module):
    """拡張アストロサイトモジュール"""
    
    def __init__(self, region_shape: Tuple[int, ...], activation_threshold: float = 0.7, 
                 decay_rate: float = 0.05, diffusion_rate: float = 0.1,
                 temporal_integration_capacity: float = 0.8,
                 layer_coupling_strength: float = 0.6):
        super().__init__()
        
        # region_shapeが単一の整数の場合、タプルに変換
        if isinstance(region_shape, int):
            region_shape = (region_shape, region_shape)
        elif len(region_shape) == 1:
            region_shape = (region_shape[0], region_shape[0])
            
        self.region_shape = region_shape
        self.activation_threshold = activation_threshold
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate
        self.temporal_integration_capacity = temporal_integration_capacity
        self.layer_coupling_strength = layer_coupling_strength
        
        # バッファの初期化（2次元テンソルとして）
        self.register_buffer('activation', torch.zeros(region_shape))
        self.register_buffer('calcium_level', torch.zeros(region_shape))
        self.register_buffer('gliotransmitter_state', torch.zeros(region_shape))
    
    def forward(self, neural_activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """順伝播処理"""
        self.update(neural_activity)
        return self.get_modulatory_effect()
        
    def update(self, neural_activity: torch.Tensor, layer_index: int = 0, delta_t: float = 1.0,
               drug_effects: Optional[Dict[str, float]] = None) -> None:
        """状態の更新"""
        # NumPy配列をPyTorchテンソルに変換
        if not isinstance(neural_activity, torch.Tensor):
            neural_activity = torch.tensor(neural_activity, dtype=torch.float32)
            
        # デバイスの一致を確保
        if neural_activity.device != self.activation.device:
            neural_activity = neural_activity.to(self.activation.device)
            
        # 活性化の更新
        self.activation = torch.where(neural_activity > self.activation_threshold,
                                    neural_activity,
                                    self.activation * (1 - self.decay_rate))
        
        # カルシウム波の伝播
        self._propagate_calcium_wave(delta_t)
        
        # グリア伝達物質の更新
        self._update_gliotransmitters()
        
        # 薬物効果の適用
        if drug_effects is not None:
            self._apply_drug_effects(drug_effects)
    
    def _propagate_calcium_wave(self, delta_t: float = 0.1):
        """カルシウム波の伝播を計算"""
        # 活性化からのカルシウム流入
        calcium_influx = self.activation * 0.1

        # パディングの適用（2次元テンソル用）
        padding = (1, 1, 1, 1)  # 上下左右にパディング
        padded = F.pad(self.calcium_level, padding, mode='constant', value=0.0)

        # 離散ラプラシアンによる拡散（2次元）
        laplacian = (
            padded[:-2, 1:-1] +  # 上
            padded[2:, 1:-1] +   # 下
            padded[1:-1, :-2] +  # 左
            padded[1:-1, 2:] -   # 右
            4 * padded[1:-1, 1:-1]  # 中心
        )

        # カルシウムレベルの更新
        diffusion = self.diffusion_rate * laplacian
        self.calcium_level = torch.clamp(
            self.calcium_level + (calcium_influx + diffusion - self.decay_rate * self.calcium_level) * delta_t,
            min=0.0
        )
    
    def _get_slice(self, tensor: torch.Tensor, dim: int, offset: int) -> torch.Tensor:
        """指定された次元のスライスを取得する

        Args:
            tensor: 入力テンソル
            dim: 次元
            offset: オフセット

        Returns:
            スライスされたテンソル
        """
        slices = [slice(1, -1) if i != dim else slice(offset, offset + self.region_shape[i])
                 for i in range(len(self.region_shape))]
        return tensor[slices]
    
    def _update_gliotransmitters(self) -> None:
        """グリア伝達物質の状態を更新する"""
        # カルシウムレベルに基づいてグリア伝達物質を生成
        self.gliotransmitter_state = torch.sigmoid(self.calcium_level - 0.5)
    
    def _apply_drug_effects(self, drug_effects: Dict[str, float]) -> None:
        """薬物効果を適用する

        Args:
            drug_effects: 薬物効果の辞書
        """
        if 'calcium_modulation' in drug_effects:
            self.calcium_level *= (1 + drug_effects['calcium_modulation'])
        if 'gliotransmitter_modulation' in drug_effects:
            self.gliotransmitter_state *= (1 + drug_effects['gliotransmitter_modulation'])
            
    def get_modulatory_effect(self) -> Dict[str, torch.Tensor]:
        """調節効果を取得する

        Returns:
            Dict[str, torch.Tensor]: 各種調節効果を含む辞書
        """
        # カルシウム波による基本的な調節効果を計算
        base_modulation = torch.sigmoid(self.calcium_level - 0.5) * 2.0 - 1.0
        
        # グリオ伝達物質の影響を加味
        synapse_modulation = base_modulation * (1.0 + self.gliotransmitter_state * 0.5)
        
        return {
            'calcium': self.calcium_level,
            'gliotransmitter': self.gliotransmitter_state,
            'activation': self.activation,
            'synapse_modulation': synapse_modulation
        }


class AdvancedNeuromodulatorSystem:
    """
    詳細な神経伝達物質システム
    様々な引用文献に基づく実装
    """
    
    def __init__(self):
        """初期化"""
        # 興奮性神経伝達物質
        self.glutamate = self._create_transmitter(
            name="Glutamate",
            baseline=0.2,
            min_level=-0.1,
            max_level=1.0,
            decay_rate=0.15,
            receptors={
                'AMPA': {'sensitivity': 0.9, 'desensitization': 0.2},
                'NMDA': {'sensitivity': 0.7, 'desensitization': 0.1},
                'mGluR': {'sensitivity': 0.6, 'desensitization': 0.05},
            }
        )
        
        # 抑制性神経伝達物質
        self.gaba = self._create_transmitter(
            name="GABA",
            baseline=0.2,
            min_level=-0.1,
            max_level=1.0,
            decay_rate=0.12,
            receptors={
                'GABA_A': {'sensitivity': 0.85, 'desensitization': 0.15},
                'GABA_B': {'sensitivity': 0.7, 'desensitization': 0.08},
            }
        )
        
        # モノアミン神経伝達物質
        self.dopamine = self._create_transmitter(
            name="Dopamine",
            baseline=0.1,
            min_level=-0.2,
            max_level=1.0,
            decay_rate=0.2,
            receptors={
                'D1': {'sensitivity': 0.8, 'desensitization': 0.15},
                'D2': {'sensitivity': 0.75, 'desensitization': 0.12},
                'D3': {'sensitivity': 0.7, 'desensitization': 0.1},
            }
        )
        
        self.serotonin = self._create_transmitter(
            name="Serotonin",
            baseline=0.1,
            min_level=-0.5,
            max_level=0.8,
            decay_rate=0.05,
            receptors={
                '5-HT1A': {'sensitivity': 0.8, 'desensitization': 0.05},
                '5-HT2A': {'sensitivity': 0.85, 'desensitization': 0.1},
                '5-HT3': {'sensitivity': 0.75, 'desensitization': 0.2},
                '5-HT7': {'sensitivity': 0.7, 'desensitization': 0.1},
            }
        )
        
        self.noradrenaline = self._create_transmitter(
            name="Noradrenaline",
            baseline=0.1,
            min_level=-0.1,
            max_level=0.9,
            decay_rate=0.15,
            receptors={
                'alpha1': {'sensitivity': 0.85, 'desensitization': 0.1},
                'alpha2': {'sensitivity': 0.7, 'desensitization': 0.15},
                'beta': {'sensitivity': 0.9, 'desensitization': 0.2},
            }
        )
        
        self.acetylcholine = self._create_transmitter(
            name="Acetylcholine",
            baseline=0.2,
            min_level=0.0,
            max_level=0.8,
            decay_rate=0.1,
            receptors={
                'nAChR': {'sensitivity': 0.85, 'desensitization': 0.3},
                'mAChR': {'sensitivity': 0.7, 'desensitization': 0.1},
            }
        )
        
        # 薬物効果の追跡
        self.active_drugs = {}
        
        # 各受容体サブタイプの不応性（脱感作状態）
        self.receptor_states = {}
        for nt in [self.glutamate, self.gaba, self.dopamine, self.serotonin, 
                  self.noradrenaline, self.acetylcholine]:
            for receptor in nt['receptors']:
                self.receptor_states[receptor] = 1.0  # 1.0は完全応答性
    
    def _create_transmitter(self, name, baseline, min_level, max_level, decay_rate, receptors):
        """神経伝達物質の基本情報を作成"""
        return {
            'name': name,
            'level': baseline,
            'baseline': baseline,
            'min_level': min_level,
            'max_level': max_level,
            'decay_rate': decay_rate,
            'receptors': receptors
        }
    
    def update(self, stimuli=None, delta_t=1.0):
        """
        神経伝達物質レベルの更新
        """
        if stimuli is None:
            stimuli = {}
        
        # 薬物効果を適用
        drug_effects = self._apply_active_drugs(delta_t)
        
        # 刺激と薬物効果を合算
        combined_effects = stimuli.copy()
        for nt, effect in drug_effects.items():
            if nt in combined_effects:
                combined_effects[nt] += effect
            else:
                combined_effects[nt] = effect
        
        # 各神経伝達物質の更新
        for nt_name, nt_obj in [('glutamate', self.glutamate), 
                               ('gaba', self.gaba),
                               ('dopamine', self.dopamine), 
                               ('serotonin', self.serotonin),
                               ('noradrenaline', self.noradrenaline), 
                               ('acetylcholine', self.acetylcholine)]:
            # ベースラインへの減衰
            decay = nt_obj['decay_rate'] * (nt_obj['level'] - nt_obj['baseline']) * delta_t
            
            # 刺激による変化
            delta = combined_effects.get(nt_name, 0.0) * delta_t
            
            # レベル更新
            nt_obj['level'] = nt_obj['level'] + delta - decay
            
            # 範囲内に収める
            nt_obj['level'] = max(nt_obj['min_level'], min(nt_obj['max_level'], nt_obj['level']))
            
            # 受容体の脱感作/回復を更新
            self._update_receptor_states(nt_obj, delta_t)
    
    def _update_receptor_states(self, nt_obj, delta_t):
        """受容体の脱感作状態を更新"""
        for receptor, properties in nt_obj['receptors'].items():
            # 現在の応答性
            current_state = self.receptor_states[receptor]
            
            # 脱感作（高レベルで脱感作が進む）
            if nt_obj['level'] > 0.6:
                desensitization = properties['desensitization'] * delta_t * (nt_obj['level'] - 0.6) / 0.4
                self.receptor_states[receptor] = max(0.2, current_state - desensitization)
            # 回復（低レベルで徐々に回復）
            else:
                recovery = 0.05 * delta_t * (1.0 - current_state)
                self.receptor_states[receptor] = min(1.0, current_state + recovery)
    
    def apply_drug(self, drug_name, dose=1.0, duration=20.0):
        """
        薬物を適用する
        Kapur & Seeman (2000)、Meltzer & Massey (2011)などの研究に基づく
        """
        drug_info = None
        
        # 抗精神病薬
        if drug_name == 'haloperidol':  # 定型抗精神病薬
            drug_info = {
                'dopamine': -0.7 * dose,        # 強いD2遮断
                'acetylcholine': -0.3 * dose,   # 抗コリン作用
                'duration': duration,
                'half_life': 24.0                # 半減期（時間）
            }
        elif drug_name == 'clozapine':  # 非定型抗精神病薬
            drug_info = {
                'dopamine': -0.4 * dose,        # 弱~中程度D2遮断
                'serotonin': -0.6 * dose,       # 強い5-HT2A遮断
                'acetylcholine': 0.2 * dose,    # ムスカリン作動性
                'noradrenaline': -0.3 * dose,   # アドレナリン遮断
                'duration': duration,
                'half_life': 12.0
            }
        
        # 抗うつ薬
        elif drug_name == 'fluoxetine':  # SSRI
            drug_info = {
                'serotonin': 0.6 * dose,        # セロトニン再取込阻害
                'duration': duration,
                'half_life': 48.0,              # 長い半減期
                'delay_effect': True            # 効果発現の遅延
            }
        elif drug_name == 'venlafaxine':  # SNRI
            drug_info = {
                'serotonin': 0.5 * dose,
                'noradrenaline': 0.5 * dose,
                'duration': duration,
                'half_life': 10.0
            }
        
        # 抗不安薬
        elif drug_name == 'diazepam':  # ベンゾジアゼピン
            drug_info = {
                'gaba': 0.7 * dose,             # GABA-A受容体の正アロステリック調節
                'duration': duration,
                'half_life': 36.0
            }
        
        # 認知増強薬
        elif drug_name == 'donepezil':  # アセチルコリンエステラーゼ阻害薬
            drug_info = {
                'acetylcholine': 0.8 * dose,
                'duration': duration,
                'half_life': 70.0
            }
        elif drug_name == 'modafinil':  # 覚醒促進薬
            drug_info = {
                'dopamine': 0.4 * dose,
                'noradrenaline': 0.5 * dose,
                'histamine': 0.3 * dose,
                'duration': duration,
                'half_life': 15.0
            }
        
        if drug_info:
            self.active_drugs[drug_name] = {
                'info': drug_info,
                'time_applied': 0.0,
                'remaining': drug_info['duration']
            }
            return True
        return False
    
    def _apply_active_drugs(self, delta_t):
        """活性化している薬物の効果を計算"""
        effects = {}
        drugs_to_remove = []
        
        for drug_name, drug_data in self.active_drugs.items():
            info = drug_data['info']
            
            # 残り時間を更新
            drug_data['time_applied'] += delta_t
            drug_data['remaining'] -= delta_t
            
            if drug_data['remaining'] <= 0:
                drugs_to_remove.append(drug_name)
                continue
            
            # 薬効の時間的減衰を計算
            if 'half_life' in info:
                decay_factor = 2 ** -(delta_t / info['half_life'])
            else:
                decay_factor = 1.0
            
            # 遅延効果（SSRIなど）の処理
            effect_multiplier = 1.0
            if info.get('delay_effect', False):
                # 効果の発現は徐々に（特に抗うつ薬）
                effect_multiplier = min(1.0, drug_data['time_applied'] / (14.0 * 24.0))  # 14日で完全効果
            
            # 各神経伝達物質への効果を適用
            for nt, effect in info.items():
                if nt not in ['duration', 'half_life', 'delay_effect']:
                    if nt in effects:
                        effects[nt] += effect * decay_factor * effect_multiplier
                    else:
                        effects[nt] = effect * decay_factor * effect_multiplier
        
        # 期限切れの薬物を削除
        for drug in drugs_to_remove:
            del self.active_drugs[drug]
        
        return effects
    
    def get_state(self):
        """神経伝達物質の状態を取得"""
        return {
            'glutamate': self.glutamate['level'],
            'gaba': self.gaba['level'],
            'dopamine': self.dopamine['level'],
            'serotonin': self.serotonin['level'],
            'noradrenaline': self.noradrenaline['level'],
            'acetylcholine': self.acetylcholine['level']
        }
    
    def get_receptor_state(self):
        """受容体の応答性状態を取得"""
        return self.receptor_states.copy()


class NeuroplasticityModule(nn.Module):
    """
    幻覚剤による神経可塑性効果をシミュレートするモジュール
    Ly et al. (2018)、Carhart-Harris & Nutt (2017)の研究に基づく
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 神経可塑性パラメータ
        self.plasticity_state = {
            'BDNF_level': 1.0,  # ベースラインレベル
            'dendritic_complexity': 1.0,
            'spine_density': 1.0,
            'synapse_strength': 1.0,
            'neurogenesis_rate': 1.0,
            'mTOR_activation': 1.0,
            'default_mode_network_activity': 1.0
        }
        
        # 持続性変化の追跡
        self.persistent_changes = {
            'structural_changes': 0.0,  # 構造的変化の蓄積
            'functional_changes': 0.0   # 機能的変化の蓄積
        }
        
        # 可塑性変化を適用するためのパラメータ
        self.plasticity_modulation = nn.Parameter(torch.ones(hidden_dim))
        self.connectivity_mask = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        
        # 神経回路の柔軟性（幻覚剤で増加）
        self.network_flexibility = nn.Parameter(torch.tensor(0.2))
        
        # BDNF/TrkBシグナリング効果
        self.trkb_signaling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def apply_psychedelic_effects(self, drug_name, dose, acute=True):
        """
        幻覚剤の神経可塑性効果を適用
        
        Args:
            drug_name: 薬物名
            dose: 用量
            acute: 急性効果か慢性効果か
        """
        if drug_name == 'lsd':
            # LSDによる効果（Ly et al., 2018）
            self.plasticity_state['BDNF_level'] += 0.8 * dose if acute else 0.3 * dose
            self.plasticity_state['dendritic_complexity'] += 0.7 * dose if acute else 0.5 * dose
            self.plasticity_state['spine_density'] += 0.6 * dose if acute else 0.4 * dose
            self.plasticity_state['default_mode_network_activity'] -= 0.8 * dose if acute else 0.2 * dose
            
            # デフォルトモードネットワーク抑制による「エントロピー増加」（Carhart-Harris et al., 2014）
            self.network_flexibility.data += 0.6 * dose * torch.tensor(1.0) if acute else 0.2 * dose * torch.tensor(1.0)
            
        elif drug_name == 'psilocybin':
            # シロシビンによる効果
            self.plasticity_state['BDNF_level'] += 0.75 * dose if acute else 0.25 * dose
            self.plasticity_state['dendritic_complexity'] += 0.6 * dose if acute else 0.4 * dose
            self.plasticity_state['spine_density'] += 0.8 * dose if acute else 0.5 * dose
            self.plasticity_state['neurogenesis_rate'] += 0.6 * dose if acute else 0.3 * dose
            self.plasticity_state['default_mode_network_activity'] -= 0.7 * dose if acute else 0.3 * dose
            
            # 神経新生効果（Catlow et al., 2013）
            self.network_flexibility.data += 0.5 * dose * torch.tensor(1.0) if acute else 0.3 * dose * torch.tensor(1.0)
            
        elif drug_name == 'ketamine':
            # ケタミンによる効果（Li et al., 2010）
            self.plasticity_state['BDNF_level'] += 0.9 * dose if acute else 0.4 * dose
            self.plasticity_state['synapse_strength'] += 0.7 * dose if acute else 0.3 * dose
            self.plasticity_state['mTOR_activation'] += 0.85 * dose if acute else 0.2 * dose
            self.plasticity_state['default_mode_network_activity'] -= 0.5 * dose if acute else 0.1 * dose
            
            # mTORシグナル活性化によるシナプス形成（Duman et al., 2016）
            self.network_flexibility.data += 0.4 * dose * torch.tensor(1.0) if acute else 0.2 * dose * torch.tensor(1.0)
        
        # 持続的変化の更新（慢性効果の蓄積）
        if not acute:
            self.persistent_changes['structural_changes'] += 0.1 * dose
            self.persistent_changes['functional_changes'] += 0.15 * dose
        
        # パラメータ正規化（過剰な値を防止）
        for key in self.plasticity_state:
            self.plasticity_state[key] = min(3.0, max(0.1, self.plasticity_state[key]))
        
        # 可塑性調整パラメータの更新
        plasticity_factor = self.plasticity_state['BDNF_level'] * 0.3 + \
                           self.plasticity_state['spine_density'] * 0.3 + \
                           self.plasticity_state['synapse_strength'] * 0.4
        
        # ニューラルネットワークのパラメータを調整
        self.plasticity_modulation.data *= (1.0 + 0.1 * (plasticity_factor - 1.0))
        
        # 結合性マスクを更新（新しい接続パターンを可能に）
        flexibility = self.network_flexibility.item()
        random_mask = torch.rand_like(self.connectivity_mask) * flexibility
        self.connectivity_mask.data = self.connectivity_mask.data * (1.0 - flexibility) + random_mask
    
    def forward(self, x):
        """
        入力表現に可塑性効果を適用
        
        Args:
            x: 入力テンソル [batch_size, hidden_dim]
            
        Returns:
            修正された表現
        """
        # BDNF/TrkBシグナリングを模倣
        bdnf_effect = self.trkb_signaling(x)
        
        # 可塑性効果の適用
        plasticity_scale = self.plasticity_modulation.unsqueeze(0).expand_as(x)
        x = x * plasticity_scale
        
        # デフォルトモードネットワーク抑制の効果（機能的統合の変化）
        dmn_factor = (2.0 - self.plasticity_state['default_mode_network_activity']) * 0.5
        
        # DMN抑制による機能的統合の増加（Carhart-Harris et al., 2016）
        # バッチサイズを取得して実際に使用
        batch_size = x.size(0)
        connectivity = self.connectivity_mask.unsqueeze(0).expand(batch_size, -1, -1)
        x_expanded = x.unsqueeze(2)
        
        # 新しい機能的結合パターン
        functional_integration = torch.bmm(x_expanded, x.unsqueeze(1)) * connectivity
        functional_integration = functional_integration.mean(dim=2) * dmn_factor
        
        # 原信号と新しい統合パターンを組み合わせ
        output = x + bdnf_effect * functional_integration
        
        return output


class NeuropharmacologicalBioKAN(nn.Module):
    def __init__(
        self,
        input_dim=784,          # MNIST入力
        hidden_dim=512,         # より大きな表現力
        output_dim=10,          # MNISTクラス数
        num_blocks=6,           # より深いネットワーク
        num_heads=8,
        dropout=0.1,
        attention_type="dot",   # 基本的な注意機構
        use_neuromodulation=False,  # 神経調節を無効化
        device="cuda"
    ):
        super().__init__()
        
        self.device = device
        
        # 入力変換層
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        
        # 注意機構
        if attention_type == "biological":
            self.attention = BiologicalMultiHeadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_neuromodulation=use_neuromodulation
            )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # BioKANブロック
        self.blocks = nn.ModuleList([
            BioKANBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_type=attention_type,
                use_neuromodulation=use_neuromodulation,
                device=device
            ) for _ in range(num_blocks)
        ])
        
        # 出力層
        self.output_transform = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # デバイスに移動
        self.to(device)

    def forward(self, x):
        # 入力の変換 [batch_size, input_dim] -> [batch_size, hidden_dim]
        h = self.input_transform(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # BioKANブロックを通す
        for block in self.blocks:
            h = block(h)
        
        # 出力層
        output = self.output_transform(h)
        return output


def create_biokan_classifier(
    in_features,
    num_classes,
    hidden_dim=None,
    num_blocks=3,
    num_heads=4,
    dropout=0.1,
    attention_type="dot",
    device="cuda",
):
    """BioKANベースの分類器を作成します。

    Args:
        in_features (int): 入力特徴量の次元
        num_classes (int): 出力クラス数
        hidden_dim (int, optional): 隠れ層の次元。Noneの場合、num_headsの倍数に調整されます。
        num_blocks (int, optional): BioKANブロックの数。デフォルトは3。
        num_heads (int, optional): マルチヘッドアテンションのヘッド数。デフォルトは4。
        dropout (float, optional): ドロップアウト率。デフォルトは0.1。
        attention_type (str, optional): アテンションのタイプ。デフォルトは"dot"。
        device (str, optional): 使用するデバイス。デフォルトは"cuda"。

    Returns:
        NeuropharmacologicalBioKAN: 作成されたモデル
    """
    if hidden_dim is None:
        # hidden_dimをnum_headsの倍数に調整
        hidden_dim = ((in_features + num_heads - 1) // num_heads) * num_heads
    else:
        # 既存のhidden_dimをnum_headsの倍数に調整
        hidden_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads

    model = NeuropharmacologicalBioKAN(
        input_dim=in_features,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=dropout,
        attention_type=attention_type,
        device=device,
    )
    
    return model
    
class PsychedelicAugmentedBioKAN(NeuropharmacologicalBioKAN):
    """
    幻覚剤効果を特に詳細にシミュレートするBioKANモデル拡張
    """
    
    def __init__(self, in_features, hidden_dim, num_classes, num_blocks=3, 
                attention_type='biological', dropout=0.1):
        super().__init__(in_features, hidden_dim, num_classes, num_blocks,
                        attention_type, dropout)
        
        # 神経可塑性モジュールを追加
        self.neuroplasticity_module = NeuroplasticityModule(hidden_dim)
        
        # 幻覚剤の視覚的効果をシミュレートするための追加層
        self.visual_processing_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(3)  # 視覚階層の簡易モデル
        ])
        
        # 薬物状態モニタリング（幻覚剤特化）
        self.psychedelic_monitoring = {
            'acute_effects': {},
            'afterglow_effects': {},
            'persistent_effects': {},
            'BDNF_timeline': [],
            'brain_network_states': [],
            'visual_processing_changes': []
        }
    
    def apply_psychedelic(self, drug_name, dose=1.0, duration=12.0, 
                          integration_sessions=0, microdosing=False):
        """
        幻覚剤を適用し、その急性効果と神経可塑性効果をシミュレート
        
        Args:
            drug_name: 薬物名（'lsd', 'psilocybin', 'ketamine'など）
            dose: 用量（標準用量を1.0とする）
            duration: 効果持続時間（時間）
            integration_sessions: 統合セッション数（持続効果に影響）
            microdosing: マイクロドーズレジメンかどうか
        """
        # マイクロドーズの場合は用量と効果を調整
        if microdosing:
            dose *= 0.1  # 通常用量の約1/10
            duration *= 0.7
            
            # マイクロドーズ特有の効果調整（Fadiman & Korb, 2019）
            self.psychedelic_monitoring['microdosing_regimen'] = True
            self.psychedelic_monitoring['microdosing_day'] = 1
        
        # 基本的な薬物効果を適用
        success = self.apply_drug(drug_name, dose, duration)
        
        if success:
            # 神経可塑性モジュールに幻覚剤効果を適用
            self.neuroplasticity_module.apply_psychedelic_effects(drug_name, dose, acute=True)
            
            # BDNF発現プロファイルのシミュレーション（Ly et al., 2018）
            bdnf_timeline = []
            for t in range(int(duration) + 1):
                if t == 0:
                    bdnf_level = 1.0  # ベースライン
                elif t < duration * 0.2:
                    bdnf_level = 1.0 + 0.5 * dose * (t / (duration * 0.2))  # 上昇期
                elif t < duration * 0.6:
                    bdnf_level = 1.0 + 0.8 * dose  # ピーク
                else:
                    bdnf_level = 1.0 + 0.8 * dose * (1.0 - (t - duration * 0.6) / (duration * 0.4))  # 下降期
                
                bdnf_timeline.append(bdnf_level)
            
            # 統合セッション効果（持続的な神経可塑性）
            afterglow_duration = duration * 1.5  # 残光効果期間
            if integration_sessions > 0:
                # 統合セッションによる持続効果の強化（Carhart-Harris et al., 2018）
                afterglow_duration *= (1.0 + 0.3 * integration_sessions)
                self.neuroplasticity_module.persistent_changes['functional_changes'] += 0.2 * integration_sessions
                
                # 内観と洞察の模倣（エピステミック変形）
                self.psychedelic_monitoring['integration_insights'] = {
                    'cognitive_flexibility': 0.3 * integration_sessions * dose,
                    'meaning_attribution': 0.4 * integration_sessions * dose,
                    'perspective_broadening': 0.5 * integration_sessions * dose
                }
            
            # デフォルトモードネットワーク抑制による「自我溶解」効果（Carhart-Harris et al., 2016）
            if dose > 0.5 and not microdosing:
                ego_dissolution_level = (dose - 0.5) * 2.0  # 0.5を超える用量で効果が現れ始める
                self.psychedelic_monitoring['acute_effects']['ego_dissolution'] = min(1.0, ego_dissolution_level)
                
                # 自我溶解の神経基盤（DMN抑制とサリエンスネットワーク活性化）
                self.psychedelic_monitoring['brain_network_states'].append({
                    'DMN_activity': 1.0 - 0.8 * dose,
                    'salience_network': 1.0 + 0.6 * dose,
                    'visual_network': 1.0 + 0.9 * dose
                })
            
            # ケタミンの抗うつ効果（Krystal et al., 2013）
            if drug_name == 'ketamine':
                self.psychedelic_monitoring['therapeutic_effects'] = {
                    'antidepressant_onset': 2.0,  # 時間
                    'antidepressant_duration': 7.0 + 2.0 * integration_sessions,  # 日
                    'antidepressant_strength': 0.7 * dose
                }
            
            # モニタリングデータを更新
            self.psychedelic_monitoring['acute_effects'][drug_name] = {
                'dose': dose,
                'duration': duration,
                'peak_bdnf': max(bdnf_timeline),
                'neuroplasticity_factor': self.neuroplasticity_module.plasticity_state['BDNF_level']
            }
            
            self.psychedelic_monitoring['BDNF_timeline'] = bdnf_timeline
            
            return True
        return False
    
    def simulate_microdosing_protocol(self, drug_name, protocol_days=30, dose=0.1,
                                      schedule="fadiman"):  # Fadiman protocol = 1 day on, 2 days off
        """
        マイクロドーズプロトコルをシミュレート（Fadiman & Korb, 2019）
        
        Args:
            drug_name: 薬物名
            protocol_days: プロトコル総日数
            dose: マイクロドーズ量（標準用量の割合）
            schedule: 投与スケジュール
        """
        results = []
        
        for day in range(1, protocol_days + 1):
            if schedule == "fadiman":
                # Fadimanプロトコル: 1日オン、2日オフ
                if day % 3 == 1:
                    self.apply_psychedelic(drug_name, dose=dose, duration=6.0, microdosing=True)
                    results.append({"day": day, "dose": dose, "effect": "on"})
                else:
                    # オフの日も脳内の変化は継続
                    self.neuroplasticity_module.apply_psychedelic_effects(drug_name, dose * 0.1, acute=False)
                    results.append({"day": day, "dose": 0, "effect": "off"})
            
            elif schedule == "stamets":
                # Stametsプロトコル: 4日オン、3日オフ
                if day % 7 <= 4:
                    self.apply_psychedelic(drug_name, dose=dose, duration=6.0, microdosing=True)
                    results.append({"day": day, "dose": dose, "effect": "on"})
                else:
                    self.neuroplasticity_module.apply_psychedelic_effects(drug_name, dose * 0.1, acute=False)
                    results.append({"day": day, "dose": 0, "effect": "off"})
        
        # 持続的変化の評価（Polito & Stevenson, 2019）
        self.psychedelic_monitoring['microdosing_results'] = {
            'mood_improvement': 0.3 + 0.2 * (protocol_days / 30),
            'cognitive_flexibility': 0.4 + 0.2 * (protocol_days / 30),
            'creative_thinking': 0.5 + 0.3 * (protocol_days / 30),
            'brain_plasticity': self.neuroplasticity_module.persistent_changes['functional_changes']
        }
        
        return results
    
    def forward(self, x):
        """
        フォワードパス（幻覚剤効果を考慮）
        """
        # 標準的な処理を実行
        h = self.input_transform(x)
        
        # BioKANブロックを通して処理
        block_outputs = []
        for block in self.blocks:
            h = block(h)
            block_outputs.append(h)
        
        # 皮質層構造の模倣
        layer_activities = self._create_cortical_layers(block_outputs)
        
        # アストロサイトの更新と効果適用
        astro_effects = self._update_astrocytes(layer_activities)
        modulated_activities = self._apply_glial_modulation(layer_activities, astro_effects)
        
        # 皮質層間の時間差ダイナミクスの処理
        h = self._process_cortical_layer_dynamics(modulated_activities)
        
        # 幻覚剤による神経可塑性効果を適用
        if self.psychedelic_monitoring['acute_effects'] or self.psychedelic_monitoring.get('microdosing_regimen', False):
            h = self.neuroplasticity_module(h)
            
            # 幻覚剤の視覚的効果をシミュレート
            if any(drug in self.psychedelic_monitoring['acute_effects'] 
                for drug in ['lsd', 'psilocybin', 'dmt']):
                
                # 視覚情報処理の変化（Kometer & Vollenweider, 2018）
                visual_changes = {}
                
                for i, layer in enumerate(self.visual_processing_layers):
                    orig_h = h.clone()
                    h = layer(h)
                    
                    # 視覚処理階層ごとの効果
                    if i == 0:  # 低次視覚特徴（色、コントラスト）
                        modulation = 1.0 + 0.5 * self._get_psychedelic_intensity()
                        h = h * modulation
                        visual_changes['low_level'] = modulation.item() if isinstance(modulation, torch.Tensor) else modulation
                        
                    elif i == 1:  # 中間視覚特徴（パターン、テクスチャ）
                        # パターン認識の変調（幾何学的視覚）
                        pattern_enhancement = self._get_psychedelic_intensity() * 0.7
                        h = h + torch.sin(h * 3.14159) * pattern_enhancement
                        visual_changes['pattern_recognition'] = pattern_enhancement
                        
                    elif i == 2:  # 高次視覚特徴（オブジェクト認識）
                        # シナプス間クロストーク（境界溶解）
                        boundary_dissolution = self._get_psychedelic_intensity() * 0.6
                        h_permuted = h[:, torch.randperm(h.size(1))]
                        h = h * (1.0 - boundary_dissolution) + h_permuted * boundary_dissolution
                        visual_changes['boundary_dissolution'] = boundary_dissolution
                
                self.psychedelic_monitoring['visual_processing_changes'].append(visual_changes)
        
        # アテンション適用
        h_expanded = h.unsqueeze(1)
        h, _ = self.attention(h_expanded, h_expanded, h_expanded)
        h = h.squeeze(1)
        
        # 神経伝達物質状態の更新
        # 活動量に基づく刺激を計算
        average_activation = h.abs().mean().item()
        max_activation = h.abs().max().item()
        
        # 神経伝達物質への刺激
        stimuli = {
            'glutamate': average_activation * 0.8,
            'gaba': (1.0 - average_activation) * 0.6,
            'dopamine': max_activation * 0.6 - 0.2,
            'acetylcholine': average_activation * 0.5,
            'noradrenaline': max_activation * 0.7 - 0.3,
            'serotonin': (1.0 - max_activation) * 0.4
        }
        
        # 幻覚剤による神経伝達物質調整
        drug_intensity = self._get_psychedelic_intensity()
        if drug_intensity > 0:
            # 5-HT2A受容体作動による神経伝達物質変化（Nichols, 2016）
            stimuli['glutamate'] += drug_intensity * 0.6  # 前頭前皮質グルタミン酸放出増加
            stimuli['dopamine'] += drug_intensity * 0.3  # 間接的ドーパミン増加
            
            # GABA介在ニューロン調節（Carhart-Harris & Nutt, 2017）
            stimuli['gaba'] -= drug_intensity * 0.4  # GABA介在ニューロン抑制
        
        # 神経伝達物質を更新
        self.neuromodulator.update(stimuli)
        
        # 神経伝達物質の効果を適用
        h = self._apply_neuromodulatory_effects(h)
        
        # 監視データの更新
        self._update_monitoring()
        
        # 出力層
        output = self.output(h)
        
        return output
    
    def _get_psychedelic_intensity(self):
        """現在の幻覚剤効果の強度を取得"""
        intensity = 0.0
        
        for drug, effects in self.psychedelic_monitoring['acute_effects'].items():
            if drug in ['lsd', 'psilocybin', 'dmt', 'ketamine']:
                intensity += effects['dose']
        
        # マイクロドーズの場合
        if self.psychedelic_monitoring.get('microdosing_regimen', False):
            if self.psychedelic_monitoring.get('microdosing_day', 0) % 3 == 1:  # 投与日
                intensity += 0.1  # マイクロドーズ効果
        
        return min(1.0, intensity)
    
class EnhancedTernaryActivation(nn.Module):
    """
    生物学的に妥当な三値活性化関数（抑制性・興奮性ニューロンの均衡を模倣）
    
    -1: 抑制性ニューロン活性化（GABAニューロン）
     0: 静止状態
     1: 興奮性ニューロン活性化（グルタミン酸ニューロン）
    """
    def __init__(self, e_i_balance=0.8, threshold=0.5, temperature=1.0):
        super().__init__()
        self.e_i_balance = e_i_balance  # 興奮性/抑制性バランス（通常80:20）
        self.threshold = threshold
        self.temperature = temperature
        
        # 三値化用の脱分極/過分極閾値
        self.hyperpolarization_threshold = -threshold  # 過分極閾値（-1に対応）
        self.depolarization_threshold = threshold      # 脱分極閾値（1に対応）
    
    def forward(self, x, neuromodulators=None):
        """
        三値活性化関数
        
        Args:
            x: 入力テンソル
            neuromodulators: 神経伝達物質状態（オプション）
            
        Returns:
            三値化された出力
        """
        # 閾値の動的調整（神経伝達物質の影響）
        threshold_mod = self.threshold
        if neuromodulators is not None:
            # ドーパミンは閾値を下げる（活性化しやすくする）
            if 'dopamine' in neuromodulators:
                threshold_mod -= 0.1 * neuromodulators['dopamine']
            
            # GABAは閾値を上げる（活性化しにくくする）
            if 'gaba' in neuromodulators:
                threshold_mod += 0.1 * neuromodulators['gaba']
        
        # E/Iバランスに基づく調整
        ei_mask = torch.rand_like(x)
        inhibitory_neurons = ei_mask > self.e_i_balance  # 抑制性ニューロン（約20%）
        
        # 訓練中は確率的三値化
        if self.training:
            # シグモイド変換で0-1に
            sigmoid_x = torch.sigmoid(x / self.temperature)
            
            # 各値の確率
            p_neg = 1.0 - torch.clamp(sigmoid_x + 0.5, 0.0, 1.0)  # -1の確率
            p_pos = torch.clamp(sigmoid_x - 0.5, 0.0, 1.0)        # 1の確率
            p_zero = 1.0 - p_neg - p_pos                         # 0の確率
            
            # 一様乱数からサンプリング
            u = torch.rand_like(x)
            ternary = torch.zeros_like(x)
            
            # 三値化
            ternary = torch.where(u < p_neg, -torch.ones_like(x), ternary)
            ternary = torch.where((u >= p_neg) & (u < p_neg + p_zero), torch.zeros_like(x), ternary)
            ternary = torch.where(u >= p_neg + p_zero, torch.ones_like(x), ternary)
            
            # 抑制性ニューロンは出力を反転（興奮入力で抑制出力）
            ternary = torch.where(inhibitory_neurons, -ternary, ternary)
            
            # ストレートスルー推定器
            ternary_no_grad = ternary.detach()
            ternary = x - x.detach() + ternary_no_grad
            
        else:
            # 評価時は決定論的三値化
            ternary = torch.zeros_like(x)
            
            # 閾値に基づく三値化
            ternary = torch.where(x < -threshold_mod, -torch.ones_like(x), ternary)
            ternary = torch.where(x > threshold_mod, torch.ones_like(x), ternary)
            
            # 抑制性ニューロンは出力を反転
            ternary = torch.where(inhibitory_neurons, -ternary, ternary)
        
        return ternary


class TernaryLayer(nn.Module):
    """
    三値ニューロンレイヤー
    """
    def __init__(self, in_features, out_features, neuromodulation=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.neuromodulation = neuromodulation
        
        # 重みとバイアス
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # 三値活性化関数
        self.activation = EnhancedTernaryActivation(e_i_balance=0.8, threshold=0.5)
        
        # E/Iバランスマスク（80%興奮性、20%抑制性）
        self.ei_mask = nn.Parameter(torch.bernoulli(torch.ones(out_features) * 0.8) * 2 - 1, 
                                   requires_grad=False)
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """神経科学的に妥当な重み初期化"""
        # Glorotの初期化（サイズに応じたスケーリング）
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
        # 抑制性ニューロンの重みは負に
        for i in range(self.out_features):
            if self.ei_mask[i] < 0:  # 抑制性ニューロン
                self.weight.data[i] = -torch.abs(self.weight.data[i])
    
    def forward(self, x, neuromodulators=None):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, in_features]
            neuromodulators: 神経伝達物質状態（オプション）
            
        Returns:
            三値活性化された出力 [batch_size, out_features]
        """
        # 線形変換
        linear_output = F.linear(x, self.weight, self.bias)
        
        # 三値活性化（神経伝達物質効果を含む）
        output = self.activation(linear_output, neuromodulators)
        
        return output


class NeuroTransformerBioKAN(nn.Module):
    """
    Transformerの強みとBioKANの神経生物学的特性を融合した拡張モデル
    """
    
    def __init__(self, 
                 in_features, 
                 hidden_dim, 
                 num_classes, 
                 num_blocks=3,
                 num_heads=8, 
                 num_layers=6,
                 dropout=0.1,
                 max_seq_length=1024,
                 use_neuroplasticity=True,
                 use_glia=True):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.use_neuroplasticity = use_neuroplasticity
        self.use_glia = use_glia
        
        # 入力の埋め込み層
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 位置エンコーディング
        self.pos_encoding = self._create_sinusoidal_encoding(max_seq_length, hidden_dim)
        
        # アテンション層
        self.attention_layers = nn.ModuleList([
            self._create_layer(hidden_dim, num_heads, dropout, i)
            for i in range(num_layers)
        ])
        
        # 層正規化
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 神経修飾システム
        self.use_neuromodulation = True
        self.neuromodulator = NeuromodulatorSystem()
        
        # グリア細胞
        if use_glia:
            self.astrocytes = nn.ModuleList([
                EnhancedAstrocyte((hidden_dim,))
                for _ in range(num_layers)
            ])
        
        # 神経可塑性モジュール
        if use_neuroplasticity:
            self.neuroplasticity = NeuroplasticityModule(hidden_dim)
        
        # 出力層
        self.output_layer = nn.Linear(hidden_dim, num_classes)
    
    def _create_sinusoidal_encoding(self, max_length, dim):
        """
        三角関数ベースの位置エンコーディング
        """
        pe = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # トレーニング可能なスケールファクター（生物学的変動性を反映）
        self.pe_scale = nn.Parameter(torch.ones(1))
        
        return pe.unsqueeze(0)
    
    def _create_layer(self, hidden_dim, num_heads, dropout, layer_index):
        """
        皮質層に特化したアテンションレイヤーを作成
        """
        if layer_index == 0:
            # 感覚入力層：高解像度・局所的コンテキスト
            return LocalBiologicalAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
        dropout=dropout,
                local_context=16,  # 局所コンテキスト窓サイズ
                neuromodulation=True
            )
        elif layer_index == 1 or layer_index == 2:
            # 浅層：水平方向結合による特徴統合
            return HorizontalIntegrationAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                neuromodulation=True
            )
        elif layer_index == 3:
            # 中間層：入力統合・中距離コンテキスト
            return BiologicalMultiHeadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                neuromodulation=True
            )
        elif layer_index == 4 or layer_index == 5:
            # 深層：広域コンテキスト・長距離依存性
            return HierarchicalMultiScaleAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_scales=3
            )
        else:  # layer 6
            # 最深層：フィードバック・トップダウン
            return FeedbackAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                neuromodulation=True
            )
    
    def forward(self, x, mask=None):
        """順伝播処理"""
        batch_size, seq_len, _ = x.size()
        
        # 位置エンコーディングの適用
        if not hasattr(self, 'pos_encoding') or self.pos_encoding.size(0) < seq_len:
            self.pos_encoding = self._create_sinusoidal_encoding(seq_len, self.hidden_dim).to(x.device)
        
        # 入力の埋め込み
        h = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        h = h + self.pos_encoding[:seq_len, :]
        h = self.dropout(h)
        
        # 各層でのアテンション処理
        for i, (attn_layer, norm_layer) in enumerate(zip(self.attention_layers, self.norm_layers)):
            # アテンション計算
            attn_out, _ = attn_layer(h)
            h = h + self.dropout(attn_out)
            h = norm_layer(h)
            
            # 神経修飾効果の適用
            if self.use_neuromodulation:
                neuromodulator_effects = self.neuromodulator.get_state()
                self.neuromodulator.update(neuromodulator_effects)
                h = self._apply_neuromodulatory_effects(h)
        
        # グリア細胞の効果を適用
        if self.use_glia:
            for astrocyte in self.astrocytes:
                h = h + astrocyte(h)["excitation"].unsqueeze(1)
        
        # 神経可塑性の効果を適用
        if self.use_neuroplasticity:
            h = self.neuroplasticity(h)
        
        # 最終的な出力層
        h = self.output_layer(h[:, -1, :])  # 最後のトークンの状態を使用
        
        return h

# 不足しているアテンションクラスを実装
class LocalBiologicalAttention(BiologicalMultiHeadAttention):
    """
    局所的な文脈を考慮したバイオロジカルアテンション
    近傍のトークンとの関係性に注目する
    """
    
    def __init__(self, 
        embed_dim, 
        num_heads, 
        local_context=16,    # 局所的な文脈に制限
        dropout=0.1
    ):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_heads: ヘッド数
            dropout: ドロップアウト率
            local_context: 局所的な文脈のウィンドウサイズ
            neuromodulation: 神経調節を行うかどうか
        """
        super().__init__(embed_dim, num_heads, dropout, True)
        self.local_context = local_context
        
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        """
        局所的な文脈を考慮した順伝播
        
        Args:
            query: クエリテンソル [batch_size, seq_len, embed_dim]
            key: キーテンソル（Noneの場合はqueryを使用）
            value: バリューテンソル（Noneの場合はqueryを使用）
            attn_mask: アテンションマスク
            need_weights: アテンション重みを返すかどうか
            
        Returns:
            attn_output: アテンション出力
            attn_weights: アテンション重み（オプション）
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, _ = query.size()
        
        # 局所的な文脈マスクを作成
        local_mask = torch.zeros(seq_len, seq_len, device=query.device)
        for i in range(seq_len):
            start = max(0, i - self.local_context // 2)
            end = min(seq_len, i + self.local_context // 2 + 1)
            local_mask[i, start:end] = 1.0
            
        # 既存のマスクと組み合わせる
        if attn_mask is not None:
            local_mask = local_mask * attn_mask
            
        # スーパークラスのforwardを呼び出し
        attn_output, attn_weights = super().forward(
            query, key, value,
            attn_mask=local_mask,
            need_weights=need_weights
        )
        
        return attn_output, attn_weights

# 残りのアテンションクラスを同様に実装
class HorizontalIntegrationAttention(BiologicalMultiHeadAttention):
    """
    水平方向の情報統合を行うバイオロジカルアテンション
    同じ層内での情報の統合を担当
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, neuromodulation=True):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_heads: ヘッド数
            dropout: ドロップアウト率
            neuromodulation: 神経調節を行うかどうか
        """
        super().__init__(embed_dim, num_heads, dropout, neuromodulation)
        
        # 水平方向の結合強度
        self.horizontal_strength = nn.Parameter(torch.ones(num_heads) * 0.5)
        
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        """
        水平方向の情報統合を行う順伝播
        
        Args:
            query: クエリテンソル [batch_size, seq_len, embed_dim]
            key: キーテンソル（Noneの場合はqueryを使用）
            value: バリューテンソル（Noneの場合はqueryを使用）
            attn_mask: アテンションマスク
            need_weights: アテンション重みを返すかどうか
            
        Returns:
            attn_output: アテンション出力
            attn_weights: アテンション重み（オプション）
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        # スーパークラスのforwardを呼び出し
        attn_output, attn_weights = super().forward(
            query, key, value,
            attn_mask=attn_mask,
            need_weights=True
        )
        
        if self.neuromodulation:
            # 水平方向の結合強度を適用
            batch_size = query.size(0)
            seq_len = query.size(1)
            
            # 結合強度を拡張 [num_heads] -> [batch_size, seq_len, embed_dim]
            strength = self.horizontal_strength.view(1, 1, -1).expand_as(attn_output)
            
            # 水平方向の結合を適用
            attn_output = attn_output * strength
            
            # 神経伝達物質の影響を考慮
            if self.neuromodulator is not None:
                # GABAの抑制効果を取得
                neuromod_effect = self.neuromodulator.get_state()
                gaba = neuromod_effect.get('gaba', 1.0)
                
                # 抑制性の調節を適用
                if gaba > 1.2:
                    attn_output = attn_output * 0.8
        
        return attn_output, attn_weights if need_weights else None

class FeedbackAttention(BiologicalMultiHeadAttention):
    """
    上位層からの情報のフィードバックを行うバイオロジカルアテンション
    予測的符号化と階層的情報処理を実現
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, neuromodulation=True):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_heads: ヘッド数
            dropout: ドロップアウト率
            neuromodulation: 神経調節を行うかどうか
        """
        super().__init__(embed_dim, num_heads, dropout, neuromodulation)
        
        # フィードバック強度の学習可能なパラメータ
        self.feedback_strength = nn.Parameter(torch.ones(num_heads) * 0.3)
        
        # 予測誤差の計算用の線形層
        self.error_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        """
        フィードバック情報を考慮した順伝播
        
        Args:
            query: クエリテンソル [batch_size, seq_len, embed_dim]
            key: キーテンソル（Noneの場合はqueryを使用）
            value: バリューテンソル（Noneの場合はqueryを使用）
            attn_mask: アテンションマスク
            need_weights: アテンション重みを返すかどうか
            
        Returns:
            attn_output: アテンション出力
            attn_weights: アテンション重み（オプション）
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        # スーパークラスのforwardを呼び出し
        attn_output, attn_weights = super().forward(
            query, key, value,
            attn_mask=attn_mask,
            need_weights=True
        )
        
        if self.neuromodulation:
            # 予測誤差の計算
            predicted = self.error_projection(attn_output)
            prediction_error = torch.abs(predicted - query)
            
            # フィードバック強度を適用
            batch_size = query.size(0)
            seq_len = query.size(1)
            
            # フィードバック強度を拡張 [num_heads] -> [batch_size, seq_len, embed_dim]
            strength = self.feedback_strength.view(1, 1, -1).expand_as(attn_output)
            
            # 予測誤差に基づくフィードバック
            feedback = attn_output * (1.0 - prediction_error * strength)
            
            # 神経伝達物質の影響を考慮
            if self.neuromodulator is not None:
                # アセチルコリンとノルアドレナリンの効果を取得
                neuromod_effect = self.neuromodulator.get_state()
                acetylcholine = neuromod_effect.get('acetylcholine', 1.0)
                noradrenaline = neuromod_effect.get('noradrenaline', 1.0)
                
                # 注意と覚醒の調整
                if acetylcholine > 1.2:
                    # アセチルコリン高：フィードバック信号を強化
                    feedback = feedback * 1.2
                
                if noradrenaline > 1.2:
                    # ノルアドレナリン高：信号対雑音比を向上
                    feedback = torch.where(
                        prediction_error < 0.3,
                        feedback * 1.2,
                        feedback * 0.8
                    )
            
            # フィードバック信号を出力に適用
            attn_output = feedback
        
        return attn_output, attn_weights if need_weights else None

class ThalamicAttention(BiologicalMultiHeadAttention):
    """
    視床様のゲーティング機構を持つバイオロジカルアテンション
    情報の選択的な伝達と注意の制御を実現
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, neuromodulation=True):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_heads: ヘッド数
            dropout: ドロップアウト率
            neuromodulation: 神経調節を行うかどうか
        """
        super().__init__(embed_dim, num_heads, dropout, neuromodulation)
        
        # ゲーティング機構
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()
        )
        
        # 状態監視
        self.state_monitor = nn.Linear(embed_dim, 1)
        
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        """
        視床様のゲーティングを適用した順伝播
        
        Args:
            query: クエリテンソル [batch_size, seq_len, embed_dim]
            key: キーテンソル（Noneの場合はqueryを使用）
            value: バリューテンソル（Noneの場合はqueryを使用）
            attn_mask: アテンションマスク
            need_weights: アテンション重みを返すかどうか
            
        Returns:
            attn_output: アテンション出力
            attn_weights: アテンション重み（オプション）
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        # スーパークラスのforwardを呼び出し
        attn_output, attn_weights = super().forward(
            query, key, value,
            attn_mask=attn_mask,
            need_weights=True
        )
        
        if self.neuromodulation:
            # 状態の監視（覚醒レベルの推定）
            arousal = torch.sigmoid(self.state_monitor(attn_output)).mean()
            
            # ゲーティング信号の生成
            gate_signal = self.gate(attn_output)
            
            # ゲーティングの適用
            gated_output = attn_output * gate_signal
            
            # 神経伝達物質の影響を考慮
            if self.neuromodulator is not None:
                # セロトニンとドーパミンの効果を取得
                neuromod_effect = self.neuromodulator.get_state()
                serotonin = neuromod_effect.get('serotonin', 1.0)
                dopamine = neuromod_effect.get('dopamine', 1.0)
                
                # 覚醒レベルに基づく調整
                if arousal > 0.8:
                    # 高覚醒時：選択性を向上
                    gate_signal = gate_signal * 1.2
                elif arousal < 0.3:
                    # 低覚醒時：全体的な抑制
                    gate_signal = gate_signal * 0.8
                
                # 報酬系の影響
                if dopamine > 1.2:
                    # 報酬関連の信号を強化
                    positive_signals = torch.relu(gated_output)
                    gated_output = gated_output + 0.2 * positive_signals
                
                # 感情状態の影響
                if serotonin > 1.2:
                    # セロトニン高：安定した情報処理
                    gated_output = torch.tanh(gated_output)
            
            # ゲーティングされた出力を返す
            attn_output = gated_output
        
        return attn_output, attn_weights if need_weights else None

class NeoCortexBioKAN(NeuroTransformerBioKAN):
    """
    人間の大脳新皮質の機能をより忠実に模倣したBioKANモデル
    
    参考文献:
    - Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
    - Sporns, O. (2011). Networks of the Brain. MIT Press.
    - Buzsáki, G. (2006). Rhythms of the Brain. Oxford University Press.
    - Goldman-Rakic, P.S. (1995). Cellular basis of working memory. Neuron, 14(3), 477-485.
    """
    
    def __init__(self, 
                 in_features, 
                 hidden_dim, 
                 num_classes, 
                 num_blocks=3,
                 num_heads=8, 
                 num_layers=6,
                 dropout=0.1,
                 max_seq_length=1024,
                 use_neuroplasticity=True,
                 use_glia=True,
                 use_working_memory=True,
                 use_predictive_coding=True,
                 oscillatory_dynamics=True):
        """
        初期化メソッド
        
        Args:
            in_features: 入力特徴量の次元
            hidden_dim: 隠れ層の次元
            num_classes: 出力クラス数
            num_blocks: BioKANブロックの数
            num_heads: マルチヘッドアテンションのヘッド数
            num_layers: レイヤー数
            dropout: ドロップアウト率
            max_seq_length: 最大シーケンス長
            use_neuroplasticity: 神経可塑性モジュールを使用するか
            use_glia: グリア細胞モジュールを使用するか
            use_working_memory: ワーキングメモリモジュールを使用するか
            use_predictive_coding: 予測符号化モジュールを使用するか
            oscillatory_dynamics: 脳波に似た振動ダイナミクスを使用するか
        """
        super().__init__(in_features, hidden_dim, num_classes, num_blocks, 
                        num_heads, num_layers, dropout, max_seq_length,
                        use_neuroplasticity, use_glia)
        
        self.use_working_memory = use_working_memory
        self.use_predictive_coding = use_predictive_coding
        self.oscillatory_dynamics = oscillatory_dynamics
        
        # 脳波パターン生成器（θ, α, β, γ, δ波）
        self.oscillation_generators = nn.ModuleDict({
            'theta': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid()
            ),
            'alpha': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid()
            ),
            'beta': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid()
            ),
            'gamma': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid()
            ),
            'delta': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid()
            )
        })
        
        # ワーキングメモリモジュール（Goldman-Rakic, 1995に基づく前頭前皮質モデル）
        if use_working_memory:
            self.working_memory = WorkingMemoryModule(hidden_dim)
        
        # 予測符号化モジュール（Friston, 2010の自由エネルギー原理に基づく）
        if use_predictive_coding:
            self.predictive_coder = PredictiveCodingModule(hidden_dim)
            
        # デフォルトモードネットワーク（自己参照処理と内部思考）
        self.default_mode_network = DefaultModeNetworkModule(hidden_dim)
        
        # 大脳皮質層構造（層I～VIの模倣）
        self.cortical_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ) for _ in range(6)  # 6層構造
        ])
        
        # 高次認知機能モジュール
        self.higher_cognition = HigherCognitionModule(hidden_dim)
    
    def forward(self, x, mask=None, context=None, memory_state=None):
        """順伝播処理
        
        Args:
            x: 入力テンソル [batch_size, seq_length, in_features]
            mask: 注意マスク
            context: コンテキスト情報
            memory_state: 外部記憶状態
            
        Returns:
            output: モデル出力
            memory_state: 更新された記憶状態
            predictive_error: 予測誤差
            attention_maps: 注意マップ
        """
        batch_size, seq_length = x.size(0), x.size(1)
        
        # 標準的なTransformerベースの処理
        h = super().forward(x, mask)
        
        # 皮質層処理（層I～VIの情報処理）
        layer_outputs = []
        cortical_input = h
        for i, layer in enumerate(self.cortical_layers):
            cortical_output = layer(cortical_input)
            # 層間の接続パターン（生物学的に妥当）
            if i > 0:
                # 上位層から下位層へのフィードバック
                cortical_output = cortical_output + 0.1 * layer_outputs[i-1]
            if i < 5:
                # 下位層への浅いスキップ接続
                cortical_input = cortical_output
            else:
                # 最終層は統合的な表現を形成
                cortical_input = cortical_output + sum(layer_outputs) * 0.05
            layer_outputs.append(cortical_output)
            
        # 脳波リズムの適用（θ, α, β, γ, δ波）
        if self.oscillatory_dynamics:
            oscillation_effects = []
            # 各周波数帯域の生成と適用
            for name, generator in self.oscillation_generators.items():
                wave = generator(cortical_output)
                if name == 'theta':
                    # θ波: 記憶と空間ナビゲーション
                    wave = wave * torch.sin(torch.linspace(0, 6*torch.pi, seq_length).to(x.device)).unsqueeze(0).unsqueeze(-1)
                elif name == 'alpha':
                    # α波: 静止状態と抑制
                    wave = wave * torch.sin(torch.linspace(0, 10*torch.pi, seq_length).to(x.device)).unsqueeze(0).unsqueeze(-1)
                elif name == 'beta':
                    # β波: 認知処理と注意
                    wave = wave * torch.sin(torch.linspace(0, 20*torch.pi, seq_length).to(x.device)).unsqueeze(0).unsqueeze(-1)
                elif name == 'gamma':
                    # γ波: 高度な認知処理と結合
                    wave = wave * torch.sin(torch.linspace(0, 40*torch.pi, seq_length).to(x.device)).unsqueeze(0).unsqueeze(-1)
                else:  # delta
                    # δ波: 深い睡眠/休息
                    wave = wave * torch.sin(torch.linspace(0, 3*torch.pi, seq_length).to(x.device)).unsqueeze(0).unsqueeze(-1)
                oscillation_effects.append(wave)
            
            # 脳波モジュレーション効果の合成
            oscillation_modulation = sum(oscillation_effects) / len(oscillation_effects)
            cortical_output = cortical_output * (1.0 + 0.2 * oscillation_modulation)
        
        # ワーキングメモリの適用
        if self.use_working_memory and memory_state is not None:
            cortical_output, memory_state = self.working_memory(cortical_output, memory_state)
        elif self.use_working_memory:
            cortical_output, memory_state = self.working_memory(cortical_output)
        
        # 予測符号化の適用
        if self.use_predictive_coding:
            cortical_output, predictive_error = self.predictive_coder(cortical_output, x)
        else:
            predictive_error = None
            
        # デフォルトモードネットワーク（自己参照的思考）の適用
        if context is not None:
            cortical_output = self.default_mode_network(cortical_output, context)
        else:
            cortical_output = self.default_mode_network(cortical_output)
            
        # 高次認知機能の適用（アナロジー、抽象思考など）
        cortical_output = self.higher_cognition(cortical_output)
        
        # 最終出力層
        output = self.output(cortical_output)
        
        if self.use_working_memory and self.use_predictive_coding:
            return output, memory_state, predictive_error, self.attention_weights
        elif self.use_working_memory:
            return output, memory_state, None, self.attention_weights
        elif self.use_predictive_coding:
            return output, None, predictive_error, self.attention_weights
        else:
            return output, None, None, self.attention_weights

class WorkingMemoryModule(nn.Module):
    """
    前頭前皮質に基づくワーキングメモリモジュール
    
    参考文献:
    - Goldman-Rakic, P.S. (1995). Cellular basis of working memory. Neuron, 14(3), 477-485.
    - Baddeley, A. (2012). Working memory: theories, models, and controversies. Annual Review of Psychology, 63, 1-29.
    """
    
    def __init__(self, hidden_dim, capacity=5, decay_factor=0.95):
        """
        初期化メソッド
        
        Args:
            hidden_dim: 隠れ層の次元
            capacity: メモリの容量（項目数）
            decay_factor: 記憶減衰係数
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.capacity = capacity
        self.decay_factor = decay_factor
        
        # ゲート機構
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        # 記憶更新機構
        self.memory_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # 注意機構
        self.attention = BiologicalMultiHeadAttention(
            hidden_dim, num_heads=4, dropout=0.1, neuromodulation=True
        )
        
    def forward(self, x, memory_state=None):
        """
        順伝播処理
        
        Args:
            x: 入力テンソル [batch_size, hidden_dim]
            memory_state: 現在のメモリ状態（Noneの場合は初期化）
            
        Returns:
            output: 処理後の出力
            updated_memory: 更新されたメモリ状態
        """
        batch_size = x.size(0)
        
        # メモリの初期化（存在しない場合）
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.capacity, self.hidden_dim, device=x.device)
        
        # 入力データとメモリの各スロットを結合して処理
        memory_inputs = []
        for i in range(self.capacity):
            memory_slot = memory_state[:, i, :]
            combined = torch.cat([x, memory_slot], dim=1)
            memory_inputs.append(combined)
            
        # 各メモリスロットに対してゲート処理
        updated_memory = []
        for i in range(self.capacity):
            combined = memory_inputs[i]
            
            # ゲート計算
            update = self.update_gate(combined)
            forget = self.forget_gate(combined)
            
            # 新しいメモリコンテンツの計算
            new_memory = self.memory_transform(combined)
            
            # メモリ更新
            slot = memory_state[:, i, :]
            updated_slot = forget * slot + update * new_memory
            
            # 減衰係数の適用（経時的な記憶衰退）
            updated_slot = updated_slot * self.decay_factor
            
            updated_memory.append(updated_slot.unsqueeze(1))
            
        # 更新されたメモリを結合
        updated_memory = torch.cat(updated_memory, dim=1)
        
        # メモリから情報を取得するための注意機構
        memory_query = x.unsqueeze(1)
        memory_output, _ = self.attention(memory_query, updated_memory, updated_memory)
        memory_output = memory_output.squeeze(1)
        
        # 最終出力（入力と記憶の組み合わせ）
        output = x + memory_output
        
        return output, updated_memory

class PredictiveCodingModule(nn.Module):
    """
    予測符号化モジュール - Fristonの自由エネルギー原理に基づく
    
    参考文献:
    - Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
    - Rao, R.P., & Ballard, D.H. (1999). Predictive coding in the visual cortex. Nature Neuroscience, 2(1), 79-87.
    """
    
    def __init__(self, hidden_dim, prediction_levels=3):
        """
        初期化メソッド
        
        Args:
            hidden_dim: 隠れ層の次元
            prediction_levels: 予測階層の数
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.prediction_levels = prediction_levels
        
        # 階層的予測ネットワーク
        self.prediction_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(prediction_levels)
        ])
        
        # 予測誤差の計算と伝播
        self.error_units = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh()
            ) for _ in range(prediction_levels)
        ])
        
        # 精度の推定（予測誤差の重み付け）
        self.precision_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Softplus()
            ) for _ in range(prediction_levels)
        ])
        
    def forward(self, x, target=None):
        """
        順伝播処理
        
        Args:
            x: 入力表現 [batch_size, hidden_dim]
            target: 予測ターゲット（指定されていない場合は入力を使用）
            
        Returns:
            output: 処理後の出力
            prediction_error: 予測誤差
        """
        if target is None:
            target = x
            
        batch_size = x.size(0)
        current_state = x
        
        # 予測誤差の蓄積
        all_prediction_errors = []
        
        # 階層的な予測処理
        for level in range(self.prediction_levels):
            # 現在の状態から予測を生成
            prediction = self.prediction_networks[level](current_state)
            
            # 予測誤差の計算
            if level == 0:
                prediction_error = target - prediction
            else:
                prediction_error = current_state - prediction
                
            # 精度の推定と適用
            precision = self.precision_estimators[level](current_state)
            weighted_error = prediction_error * precision
            
            # 誤差ユニットによる処理
            processed_error = self.error_units[level](weighted_error)
            all_prediction_errors.append(processed_error)
            
            # 次の階層への状態更新
            current_state = current_state + 0.1 * processed_error
        
        # 予測誤差の合計
        total_prediction_error = sum(all_prediction_errors)
        
        # 最終出力（最上位の状態と予測誤差の組み合わせ）
        output = current_state + 0.05 * total_prediction_error
        
        return output, total_prediction_error

class DefaultModeNetworkModule(nn.Module):
    """
    デフォルトモードネットワークモジュール - 自己参照的思考と内省を模倣
    
    参考文献:
    - Raichle, M.E., et al. (2001). A default mode of brain function. PNAS, 98(2), 676-682.
    - Buckner, R.L., et al. (2008). The brain's default network. Annals of the New York Academy of Sciences, 1124(1), 1-38.
    """
    
    def __init__(self, hidden_dim):
        """
        初期化メソッド
        
        Args:
            hidden_dim: 隠れ層の次元
        """
        super().__init__()
        
        # 自己参照モジュール
        self.self_reference = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # エピソード記憶の集積
        self.episodic_memory = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 心的時間旅行（過去と未来の思考）
        self.mental_time_travel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 社会的認知
        self.social_cognition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, context=None):
        """
        順伝播処理
        
        Args:
            x: 入力表現 [batch_size, hidden_dim]
            context: コンテキスト情報（オプション）
            
        Returns:
            output: 処理後の出力
        """
        # 自己参照処理
        self_ref = self.self_reference(x)
        
        # エピソード記憶の統合
        if context is not None:
            episodic = self.episodic_memory(torch.cat([x, context], dim=1))
        else:
            # コンテキストがない場合、自己生成したコンテキストを使用
            temp_context = self.mental_time_travel(x)
            episodic = self.episodic_memory(torch.cat([x, temp_context], dim=1))
        
        # 心的時間旅行
        time_travel = self.mental_time_travel(x)
        
        # 社会的認知
        social = self.social_cognition(x)
        
        # 全ての処理を統合
        output = x + 0.2 * self_ref + 0.2 * episodic + 0.1 * time_travel + 0.1 * social
        
        return output

class HigherCognitionModule(nn.Module):
    """
    高次認知機能モジュール - 抽象思考、アナロジー、推論など
    
    参考文献:
    - Lake, B.M., et al. (2017). Building machines that learn and think like people. Behavioral and Brain Sciences, 40, e253.
    - Lenat, D.B. (1995). CYC: A large-scale investment in knowledge infrastructure. Communications of the ACM, 38(11), 33-38.
    """
    
    def __init__(self, hidden_dim):
        """
        初期化メソッド
        
        Args:
            hidden_dim: 隠れ層の次元
        """
        super().__init__()
        
        # 抽象思考
        self.abstraction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # アナロジー推論
        self.analogy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 因果推論
        self.causal_reasoning = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # メタ認知
        self.metacognition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 創造性
        self.creativity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.3),  # ランダム性を導入
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x):
        """
        順伝播処理
        
        Args:
            x: 入力表現 [batch_size, hidden_dim]
            
        Returns:
            output: 処理後の出力
        """
        # 抽象思考
        abstract = self.abstraction(x)
        
        # アナロジー推論
        analog = self.analogy(x)
        
        # 因果推論
        causal = self.causal_reasoning(x)
        
        # メタ認知（自己の思考についての思考）
        meta = self.metacognition(x)
        
        # 創造性
        creative = self.creativity(x)
        
        # 全ての高次認知機能を統合
        output = x + 0.15 * abstract + 0.15 * analog + 0.1 * causal + 0.1 * meta + 0.1 * creative
        
        return output
    
    def _update_neuromodulators(self, layer_activities):
        """神経伝達物質の状態を更新
        
        Args:
            layer_activities: 各層の活性化状態
        """
        # 活性化レベルの計算
        average_activation = layer_activities[-1].mean().item()
        max_activation = layer_activities[-1].max().item()
        attention_level = self.attention_weights.mean().item() if hasattr(self, 'attention_weights') else 0.5
        arousal_level = max(0.2, min(0.8, average_activation))
        stability_level = 1.0 - torch.var(layer_activities[-1]).item()
        
        # 神経伝達物質への刺激を設定
        stimuli = {
            'glutamate': average_activation * 0.8,
            'gaba': (1.0 - average_activation) * 0.6,
            'dopamine': max_activation * 0.6 - 0.2,
            'acetylcholine': attention_level * 0.7,
            'noradrenaline': arousal_level * 0.8,
            'serotonin': stability_level * 0.5
        }
        
        # 神経伝達物質の状態を更新
        if self.neuromodulator is not None:
            self.neuromodulator.update(stimuli)
    
class NeuromodulationSystem(nn.Module):
    def __init__(self, hidden_dim, device="cuda"):
        super().__init__()
        
        self.device = device
        self.hidden_dim = hidden_dim
        
        # 神経調節パラメータ
        self.dopamine = nn.Parameter(torch.ones(1, device=device))
        self.serotonin = nn.Parameter(torch.ones(1, device=device))
        self.norepinephrine = nn.Parameter(torch.ones(1, device=device))
        self.acetylcholine = nn.Parameter(torch.ones(1, device=device))
        
        # 神経調節の変換層
        self.modulation_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        ).to(device)
        
        # 初期状態
        self.baseline_levels = {
            'dopamine': 1.0,
            'serotonin': 1.0,
            'norepinephrine': 1.0,
            'acetylcholine': 1.0
        }
        
        # デバイスに移動
        self.to(device)

    def forward(self, x):
        # 入力の形状を取得
        batch_size = x.size(0)
        
        # 神経調節の計算
        modulation = self.modulation_transform(x)
        
        # 各神経伝達物質の効果を適用
        dopamine_effect = self.dopamine * modulation
        serotonin_effect = self.serotonin * modulation
        norepinephrine_effect = self.norepinephrine * modulation
        acetylcholine_effect = self.acetylcholine * modulation
        
        # 全ての効果を組み合わせ
        combined_effect = (
            dopamine_effect +
            serotonin_effect +
            norepinephrine_effect +
            acetylcholine_effect
        ) / 4.0
        
        return combined_effect

    def update(self, stimuli):
        """
        外部刺激に基づいて神経伝達物質レベルを更新
        
        Args:
            stimuli (dict): 各神経伝達物質への刺激レベル
        """
        if 'dopamine' in stimuli:
            self.dopamine.data = torch.tensor([stimuli['dopamine']], device=self.device)
        if 'serotonin' in stimuli:
            self.serotonin.data = torch.tensor([stimuli['serotonin']], device=self.device)
        if 'norepinephrine' in stimuli:
            self.norepinephrine.data = torch.tensor([stimuli['norepinephrine']], device=self.device)
        if 'acetylcholine' in stimuli:
            self.acetylcholine.data = torch.tensor([stimuli['acetylcholine']], device=self.device)

    def get_state(self):
        """現在の神経伝達物質レベルを取得"""
        return {
            'dopamine': self.dopamine.item(),
            'serotonin': self.serotonin.item(),
            'norepinephrine': self.norepinephrine.item(),
            'acetylcholine': self.acetylcholine.item()
        }

def monitor_memory():
    print(f'GPU使用メモリ: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    print(f'GPU最大メモリ: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')

class AdvancedMNISTExperiment:
    def __init__(self, config):
        self.model = NeuropharmacologicalBioKAN(
            hidden_dim=512,
            num_blocks=6,
            attention_type="biological"
        )
        
        # 学習率スケジューラの追加
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=2e-4,
            epochs=config.num_epochs,
            steps_per_epoch=len(self.train_loader)
        )
        
        # 並列処理の最適化
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
