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


class BiologicalMultiHeadAttention(MultiHeadAttention):
    """神経学的特性を持つマルチヘッドアテンション層"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, neuromodulation=True):
        """
        初期化
        
        Args:
            embed_dim: 埋め込み次元
            num_heads: ヘッド数
            dropout: ドロップアウト確率
            neuromodulation: 神経調節を有効にするかどうか
        """
        super().__init__(embed_dim, num_heads, dropout)
        
        # 神経調節システム（オプション）
        self.neuromodulator = NeuromodulatorSystem() if neuromodulation else None
        
        # アストロサイト活動（オプション）
        if neuromodulation:
            self.astrocyte = Astrocyte(region_shape=(num_heads, embed_dim // num_heads))
        else:
            self.astrocyte = None
        
        # ヘッドごとの活性化閾値（これはニューロンの発火閾値に相当）
        self.head_thresholds = nn.Parameter(torch.ones(num_heads) * 0.1)
        
        # アテンション重みキャッシュ（説明可能性のため）
        self.attention_weights = None
    
    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        """
        フォワードパス
        
        Args:
            query: クエリテンソル
            key: キーテンソル
            value: バリューテンソル
            attn_mask: アテンションマスク
            need_weights: 重みを返すかどうか
            
        Returns:
            出力テンソルとオプションのアテンション重み
        """
        # 基本的なアテンション計算
        attn_output, attn_weights = super().forward(query, key, value, attn_mask, True)
        
        # アテンション重みをキャッシュ
        self.attention_weights = attn_weights.detach()
        
        # 神経調節システムがある場合
        if self.neuromodulator is not None:
            # アテンション活動に基づく神経調節の更新
            attn_activity = attn_weights.mean(dim=1).mean(dim=0)  # ヘッドごとの平均アテンション
            
            # 神経伝達物質レベルの更新
            neuromodulator_effects = {
                'dopamine': 0.2 * attn_activity.max().item(),       # 最大アテンションに応じた報酬
                'acetylcholine': 0.1 * attn_activity.mean().item(), # 平均アテンションに応じた注意
                'serotonin': -0.05 * (attn_weights.var().item()),   # アテンションのばらつきが大きいと抑制的
                'noradrenaline': 0.15 if attn_activity.max().item() > 0.7 else -0.05  # 強いアテンションで覚醒
            }
            
            self.neuromodulator.update(stimuli=neuromodulator_effects)
            
            # 現在の神経伝達物質状態を取得
            neuro_state = self.neuromodulator.get_state()
            
            # アテンション出力の調整
            dopamine_effect = 1.0 + 0.2 * neuro_state['dopamine']  # ドーパミンによる信号増幅
            serotonin_effect = 1.0 - 0.1 * neuro_state['serotonin'] if neuro_state['serotonin'] > 0 else 1.0 + 0.05 * abs(neuro_state['serotonin'])
            
            # ドーパミンとセロトニンによる出力調整
            attn_output = attn_output * dopamine_effect * serotonin_effect
            
            # アストロサイトの更新
            if self.astrocyte is not None:
                # アテンションをニューロン活性として解釈
                neural_activity = torch.reshape(attn_output[0], self.astrocyte.region_shape).detach().cpu().numpy()
                
                # アストロサイト状態の更新
                self.astrocyte.update(neural_activity)
                
                # アストロサイトの調節効果を取得
                astro_effects = self.astrocyte.get_modulatory_effect()
                
                # グルタミン酸・GABA取り込みの効果をアテンションに適用
                # （テンソルに変換して形状を合わせる）
                glutamate_uptake = torch.tensor(astro_effects['glutamate_uptake'], device=attn_output.device)
                synapse_mod = torch.tensor(astro_effects['synapse_modulation'], device=attn_output.device)
                
                # アストロサイトの効果を適用（次元を合わせる必要あり）
                glutamate_uptake = glutamate_uptake.view(*self.astrocyte.region_shape, 1).expand(-1, -1, attn_output.size(1))
                synapse_mod = synapse_mod.view(*self.astrocyte.region_shape, 1).expand(-1, -1, attn_output.size(1))
                
                # 次元を合わせる
                glutamate_uptake = glutamate_uptake.reshape(attn_output.shape)
                synapse_mod = synapse_mod.reshape(attn_output.shape)
                
                # アストロサイト効果の適用
                attn_output = attn_output * glutamate_uptake * synapse_mod
        
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output


class BioKANBlock(nn.Module):
    """BioKANの基本構成ブロック"""
    
    def __init__(self, in_features, hidden_dim, out_features, 
                n_layers=2, activation='tanh', use_bias=True, neuromodulation=True):
        """
        初期化
        
        Args:
            in_features: 入力特徴量数
            hidden_dim: 隠れ層の次元
            out_features: 出力特徴量数
            n_layers: 層数
            activation: 活性化関数
            use_bias: バイアスを使用するかどうか
            neuromodulation: 神経調節を有効にするかどうか
        """
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.n_layers = n_layers
        
        # 三値活性化関数
        if activation == 'ternary':
            self.activation = TernaryActivationFunction()
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = getattr(torch, activation)
        
        # レイヤーを構築
        layers = []
        current_dim = in_features
        
        for i in range(n_layers - 1):
            layers.append(KANLinear(current_dim, hidden_dim, use_bias=use_bias))
            current_dim = hidden_dim
        
        layers.append(KANLinear(current_dim, out_features, use_bias=use_bias))
        
        self.layers = nn.ModuleList(layers)
        
        # 神経調節システム（オプション）
        self.neuromodulator = NeuromodulatorSystem() if neuromodulation else None
        
        # バイオロジカルアテンション（特徴間）
        self.feature_attention = BiologicalMultiHeadAttention(
            out_features, num_heads=4, neuromodulation=neuromodulation
        )
    
    def forward(self, x):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, in_features]
            
        Returns:
            出力テンソル [batch_size, out_features]
        """
        # フィード前向き経路
        h = x
        hidden_states = [h]
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            # 中間層では活性化関数を適用
            if i < len(self.layers) - 1:
                h = self.activation(h)
                
            hidden_states.append(h)
        
        # 神経調節の影響を適用（もし有効なら）
        if self.neuromodulator is not None:
            # 活性化パターンに基づいて神経調節システムを更新
            h_abs = torch.abs(h)
            activity_level = h_abs.mean().item()
            sparsity = (h == 0).float().mean().item()
            
            # 神経伝達物質への刺激を計算
            stimuli = {
                'dopamine': 0.1 * activity_level - 0.05 * sparsity,  # 活動が高く、スパース性が低いとドーパミン放出
                'acetylcholine': 0.2 * (1 - sparsity),               # スパース性が低いとアセチルコリン放出
                'glutamate': 0.15 * activity_level,                  # 活動レベルに比例してグルタミン酸放出
                'gaba': 0.1 * sparsity                               # スパース性に比例してGABA放出
            }
            
            self.neuromodulator.update(stimuli=stimuli)
            
            # 神経伝達物質の状態を取得
            neuro_state = self.neuromodulator.get_state()
            
            # 出力の調整
            h = h * (1.0 + 0.2 * neuro_state['dopamine'])  # ドーパミンによる出力の増幅
            
            # ノルアドレナリンによる注意調整（閾値変更）
            attention_threshold = 0.5 - 0.3 * neuro_state['noradrenaline']
            h = torch.where(torch.abs(h) > attention_threshold, h, torch.zeros_like(h))
        
        # フィーチャーアテンション（出力の特徴間関係をモデル化）
        # アテンションを適用するために次元を追加して変換
        h_attn = h.unsqueeze(1)  # [batch_size, 1, out_features]
        
        # セルフアテンションとして適用
        h_attn, _ = self.feature_attention(h_attn, h_attn, h_attn)
        
        # 元の次元に戻す
        h = h_attn.squeeze(1)  # [batch_size, out_features]
        
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


class EnhancedAstrocyte:
    """
    拡張アストロサイト実装
    Cornell-Bell et al. (1990)とVerkhratsky & Nedergaard (2018)の研究に基づく
    """
    
    def __init__(self, region_shape: Tuple[int, ...], activation_threshold: float = 0.7, 
                 decay_rate: float = 0.05, diffusion_rate: float = 0.1,
                 temporal_integration_capacity: float = 0.8,
                 layer_coupling_strength: float = 0.6):
        """
        初期化
        """
        self.region_shape = region_shape
        self.activation_threshold = activation_threshold
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate
        self.temporal_integration_capacity = temporal_integration_capacity
        self.layer_coupling_strength = layer_coupling_strength
        
        # 基本状態変数
        self.activation = np.zeros(region_shape)  # Ca2+活性化状態
        self.glutamate_level = np.zeros(region_shape)
        self.gaba_level = np.zeros(region_shape)
        self.trophic_factors = np.zeros(region_shape)  # BDNF、GDNFなど
        
        # 時間差処理のための変数
        self.temporal_delay_buffer = []
        self.buffer_max_size = 5
        self.layer_specific_delays = np.linspace(0.1, 1.0, 6)  # 皮質6層の特有の時間遅延
        self.cross_layer_modulation = np.zeros(region_shape)
        
        # 新規：グリオトランスミッター放出状態（Araque et al., 2014）
        self.gliotransmitters = {
            'glutamate': np.zeros(region_shape),  # 興奮性
            'ATP': np.zeros(region_shape),       # プリン作動性
            'D-serine': np.zeros(region_shape),  # NMDA補助因子
            'TNF-alpha': np.zeros(region_shape)  # 免疫調節
        }
        
        # 新規：カルシウム波動力学パラメータ（Cornell-Bell et al., 1990）
        self.calcium_wave_velocity = 15.0  # μm/s
        self.calcium_oscillation_frequency = 0.1  # Hz
        self.ip3_sensitivity = 0.75  # IP3受容体感受性
        
        # 新規：細胞外K+緩衝能（Verkhratsky & Nedergaard, 2018）
        self.k_buffering_capacity = 0.8
    
    def update(self, neural_activity: np.ndarray, layer_index: int = 0, delta_t: float = 1.0,
               drug_effects: Optional[Dict[str, float]] = None):
        """
        アストロサイトの状態を更新（薬物効果を含む）
        """
        # 神経活動に基づく活性化
        activation_input = np.where(neural_activity > self.activation_threshold, 
                                   neural_activity, 0)
        self.activation += activation_input * delta_t
        
        # カルシウムウェーブの拡散
        self._propagate_calcium_wave(delta_t)
        
        # 神経伝達物質レベルの更新
        self.glutamate_level = np.maximum(0, neural_activity * 0.8 - self.activation * 0.4)
        self.gaba_level = np.maximum(0, -neural_activity * 0.6 + self.activation * 0.3)
        
        # 栄養因子の更新
        self.trophic_factors = self.activation * 0.5 * np.maximum(0, 1 - np.abs(neural_activity))
        
        # グリオトランスミッター放出の更新（Araque et al., 2014）
        self._update_gliotransmitters()
        
        # 皮質層間時間差処理
        self._process_temporal_cortical_layers(neural_activity, layer_index, delta_t)
        
        # 薬物効果の適用（あれば）
        if drug_effects:
            self._apply_drug_effects(drug_effects)
    
    def _propagate_calcium_wave(self, delta_t: float):
        """
        カルシウム波の伝播をシミュレート（Cornell-Bell et al., 1990）
        """
        # 畳み込みフィルタを使った拡散
        kernel = np.array([[0.05, 0.1, 0.05], 
                          [0.1, 0.4, 0.1], 
                          [0.05, 0.1, 0.05]])
        
        padded = np.pad(self.activation, 1, mode='constant')
        diffused = np.zeros_like(self.activation)
        
        # 2D拡散
        if len(self.region_shape) == 2:
            for i in range(self.region_shape[0]):
                for j in range(self.region_shape[1]):
                    diffused[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        
        # 拡散と減衰の適用
        wave_distance = self.calcium_wave_velocity * delta_t
        diffusion_scale = min(1.0, wave_distance / np.max(self.region_shape))
        
        self.activation = (1 - self.diffusion_rate * diffusion_scale) * self.activation + \
                          self.diffusion_rate * diffusion_scale * diffused
        self.activation *= (1 - self.decay_rate * delta_t)
        
        # IP3依存性カルシウム振動の追加（Volterra & Meldolesi, 2005）
        oscillation = 0.2 * np.sin(2 * np.pi * self.calcium_oscillation_frequency * delta_t)
        self.activation += oscillation * (self.activation > 0.2) * self.ip3_sensitivity
    
    def _update_gliotransmitters(self):
        """
        カルシウム濃度に基づくグリオトランスミッター放出の更新（Araque et al., 2014）
        """
        # グルタミン酸放出（高Ca2+での放出）
        self.gliotransmitters['glutamate'] = np.where(
            self.activation > 0.7,
            self.activation * 0.6,
            0
        )
        
        # ATP放出（中程度Ca2+での放出）
        self.gliotransmitters['ATP'] = np.where(
            (self.activation > 0.4) & (self.activation < 0.8),
            self.activation * 0.5,
            0
        )
        
        # D-serine放出（持続的なCa2+上昇に依存）
        self.gliotransmitters['D-serine'] = np.where(
            self.activation > 0.6,
            self.activation * 0.7,
            0
        )
        
        # TNF-alpha放出（長期的な活性化）
        # 実装省略（長期的なダイナミクスに依存）
    
    def _apply_drug_effects(self, drug_effects: Dict[str, float]):
        """
        薬物効果をアストロサイト機能に適用
        """
        # 抗精神病薬効果（Khoruzhenko et al., 2019）
        if 'antipsychotic' in drug_effects:
            dose = drug_effects['antipsychotic']
            # D2受容体阻害効果によるCa2+シグナリング変化
            self.activation *= (1.0 - 0.3 * dose)
            self.ip3_sensitivity *= (1.0 - 0.2 * dose)
        
        # 抗うつ薬効果（Duman & Monteggia, 2006）
        if 'antidepressant' in drug_effects:
            dose = drug_effects['antidepressant']
            duration = drug_effects.get('treatment_duration', 1)
            
            # 慢性効果（BDNF増加）
            chronic_factor = min(1.0, duration / 14.0)
            self.trophic_factors += 0.3 * dose * chronic_factor
            
            # セロトニンによる間接効果
            if drug_effects.get('serotonergic', False):
                # 5-HT2A受容体を介したCa2+放出
                self.activation += 0.2 * dose * (np.random.rand(*self.region_shape) < 0.3)
        
        # 認知増強薬効果
        if 'cognitive_enhancer' in drug_effects:
            dose = drug_effects['cognitive_enhancer']
            # アセチルコリン作用増強
            if drug_effects.get('cholinergic', False):
                self.temporal_integration_capacity *= (1.0 + 0.2 * dose)
                self.glutamate_level *= (1.0 - 0.2 * dose)  # グルタミン酸取り込み増加
    
    def get_modulatory_effect(self) -> Dict[str, np.ndarray]:
        """
        アストロサイトの調節効果を取得
        """
        return {
            'glutamate_uptake': 1.0 - 0.8 * self.glutamate_level,
            'gaba_uptake': 1.0 - 0.7 * self.gaba_level,
            'synapse_modulation': 0.8 + 0.4 * self.activation,
            'trophic_support': self.trophic_factors,
            'cross_layer_temporal_modulation': self.cross_layer_modulation,
            # 新規：グリオトランスミッター効果
            'glutamate_gliotransmission': self.gliotransmitters['glutamate'],
            'ATP_signaling': self.gliotransmitters['ATP'],
            'NMDA_coactivation': self.gliotransmitters['D-serine']
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
    """
    薬理学的効果をシミュレート可能なBioKANモデル
    各種引用文献に基づく実装
    """
    
    def __init__(self, in_features, hidden_dim, num_classes, num_blocks=3, 
                 attention_type='biological', dropout=0.1):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 入力変換層
        self.input_transform = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # BioKANブロック
        self.blocks = nn.ModuleList([
            BioKANBlock(
                in_features=hidden_dim if i==0 else hidden_dim,
                hidden_dim=hidden_dim,
                out_features=hidden_dim,
                neuromodulation=True
            ) for i in range(num_blocks)
        ])
        
        # アテンションメカニズム
        if attention_type == 'biological':
            self.attention = BiologicalMultiHeadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                neuromodulation=True
            )
        elif attention_type == 'cortical':
            self.attention = CorticalAttention(
                embed_dim=hidden_dim,
                num_regions=4,
                region_heads=2,
                dropout=dropout
            )
        else:  # hierarchical
            self.attention = HierarchicalMultiScaleAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                num_scales=3
            )
        
        # 出力層
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 拡張神経伝達物質システム
        self.neuromodulator = AdvancedNeuromodulatorSystem()
        
        # 拡張グリア細胞システム
        # 皮質6層構造を模倣（Mountcastle, 1997）
        self.astrocytes = nn.ModuleList([
            EnhancedAstrocyte(region_shape=(hidden_dim // 16, 16))
            for _ in range(6)  # Layer I, II/III, IV, V, VI, VIb
        ])
        
        # 層間接続（皮質マイクロカラム構造を模倣）
        self.layer_weights = nn.Parameter(torch.ones(6) / 6)
        self.temporal_importance = nn.Parameter(torch.tensor(0.7))
        self.cortical_layer_connections = nn.Parameter(torch.randn(6, 6) * 0.1)
        
        # 皮質層間活動履歴（Fields et al., 2014）
        self.cortical_activity_history = [[] for _ in range(6)]
        self.history_max_length = 10
        
        # 薬理学モニタリング
        self.drug_monitoring = {
            'applied_drugs': [],
            'nt_history': [],
            'receptor_history': [],
            'astrocyte_history': []
        }
        
        # 神経可塑性モジュールを追加
        self.neuroplasticity_module = NeuroplasticityModule(hidden_dim)
    
    def apply_drug(self, drug_name, dose=1.0, duration=20.0):
        """
        モデルに薬物を適用し、その効果をシミュレート
        """
        success = self.neuromodulator.apply_drug(drug_name, dose, duration)
        
        if success:
            self.drug_monitoring['applied_drugs'].append({
                'drug': drug_name,
                'dose': dose,
                'time_applied': len(self.drug_monitoring['nt_history']),
                'duration': duration
            })
            
            # 薬物の特性に基づいてアストロサイトにも効果を適用
            drug_effects = {}
            
            # 薬物タイプと効果のマッピング
            if drug_name in ['haloperidol', 'clozapine', 'risperidone']:
                drug_effects['antipsychotic'] = dose
            elif drug_name in ['fluoxetine', 'venlafaxine', 'sertraline']:
                drug_effects['antidepressant'] = dose
                drug_effects['serotonergic'] = True
                drug_effects['treatment_duration'] = 1  # 初期値は急性効果
            elif drug_name in ['diazepam', 'alprazolam']:
                drug_effects['anxiolytic'] = dose
            elif drug_name in ['donepezil', 'modafinil']:
                drug_effects['cognitive_enhancer'] = dose
                if drug_name == 'donepezil':
                    drug_effects['cholinergic'] = True
            
            # アストロサイトに薬物効果を適用
            for astrocyte in self.astrocytes:
                astrocyte._apply_drug_effects(drug_effects)
            
            # 神経可塑性効果を適用
            self.neuroplasticity_module.apply_psychedelic_effects(drug_name, dose)
        
        return success
    
    def set_chronic_treatment(self, duration_days):
        """
        慢性投与効果をシミュレート（Duman & Monteggia, 2006）
        """
        for drug_data in self.drug_monitoring['applied_drugs']:
            if drug_data['drug'] in ['fluoxetine', 'venlafaxine', 'sertraline']:
                # アストロサイトへの慢性効果を適用
                drug_effects = {
                    'antidepressant': drug_data['dose'],
                    'serotonergic': True,
                    'treatment_duration': duration_days
                }
                
                # BDNF増加（神経可塑性への効果）
                for astrocyte in self.astrocytes:
                    astrocyte._apply_drug_effects(drug_effects)
    
    def forward(self, x):
        """
        フォワードパス
        """
        # 入力変換
        h = self.input_transform(x)
        
        # BioKANブロックを通して処理
        block_outputs = []
        for block in self.blocks:
            h = block(h)
            block_outputs.append(h)
        
        # 皮質層構造の模倣（6層構造）
        layer_activities = self._create_cortical_layers(block_outputs)
        
        # アストロサイトの更新と効果適用
        astro_effects = self._update_astrocytes(layer_activities)
        modulated_activities = self._apply_glial_modulation(layer_activities, astro_effects)
        
        # 皮質層間の時間差ダイナミクスの処理
        h = self._process_cortical_layer_dynamics(modulated_activities)
        
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
        
        # 神経伝達物質を更新
        self.neuromodulator.update(stimuli)
        
        # 神経伝達物質の効果を適用
        h = self._apply_neuromodulatory_effects(h)
        
        # 監視データの更新
        self._update_monitoring()
        
        # 出力層
        output = self.classifier(h)
        
        return output
    
    def _create_cortical_layers(self, block_outputs):
        """
        BioKANブロック出力から皮質6層構造を作成
        Mountcastle (1997)の皮質構造に基づく
        """
        num_blocks = len(block_outputs)
        layer_activities = []
        
        if num_blocks >= 3:
            # 複数のブロック出力を6層に割り当て
            blocks_per_layer = max(1, num_blocks // 6)
            remaining_layers = 6 - (num_blocks // blocks_per_layer)
            
            for i, block_output in enumerate(block_outputs):
                for j in range(blocks_per_layer):
                    if len(layer_activities) < 6:
                        # 非線形変換で異なる表現を作成
                        layer_h = torch.tanh(block_output + 0.1 * j)
                        layer_activities.append(layer_h)
            
            # 残りの層を最後のブロックから生成
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
            
            # 残りの層を最後のブロックから生成
            while len(layer_activities) < 6:
                j = len(layer_activities) - (6 // num_blocks) * (num_blocks - 1)
                layer_h = torch.relu(block_outputs[-1] + 0.05 * j)
                layer_activities.append(layer_h)
        
        return layer_activities
    
    def _update_astrocytes(self, layer_activities):
        """
        アストロサイト層の更新と効果取得
        Cornell-Bell et al. (1990)とVerkhratsky & Nedergaard (2018)の研究に基づく
        """
        astro_effects = []
        
        for i, (activity, astrocyte) in enumerate(zip(layer_activities, self.astrocytes)):
            # バッチの最初のサンプルでアストロサイトを更新
            activity_sample = activity[0].detach().cpu().numpy()
            
            # 活動を2D形状にリシェイプ
            activity_reshaped = activity_sample.reshape(astrocyte.region_shape)
            
            # 層インデックスとともに更新
            astrocyte.update(activity_reshaped, layer_index=i)
            
            # 調節効果を取得
            modulatory_effect = astrocyte.get_modulatory_effect()
            astro_effects.append(modulatory_effect)
        
        return astro_effects
    
    def _apply_glial_modulation(self, layer_activities, astro_effects):
        """
        グリア細胞による調節を適用
        """
        modulated_activities = []
        
        for i, (activity, effect) in enumerate(zip(layer_activities, astro_effects)):
            # アストロサイトの変調効果をテンソルに変換
            synapse_mod = torch.tensor(
                effect['synapse_modulation'], 
                device=activity.device
            ).float()
            
            temporal_mod = torch.tensor(
                effect['cross_layer_temporal_modulation'],
                device=activity.device
            ).float()
            
            # 効果を拡張してバッチ次元と一致させる
            synapse_mod = synapse_mod.reshape(1, -1).expand(activity.shape[0], -1)
            temporal_mod = temporal_mod.reshape(1, -1).expand(activity.shape[0], -1)
            
            # 活動に調節効果を適用
            modulated = activity * (1.0 + 0.3 * synapse_mod)  # シナプス調節
            modulated = modulated + 0.2 * temporal_mod  # 時間差調節を加算
            
            modulated_activities.append(modulated)
        
        return modulated_activities
    
    def _process_cortical_layer_dynamics(self, layer_activities):
        """
        皮質層間のダイナミクスを処理
        
        Args:
            layer_activities: 各皮質層の活動 [6 x batch_size x hidden_dim]
            
        Returns:
            統合された活動 [batch_size x hidden_dim]
        """
        batch_size = layer_activities[0].shape[0]
        hidden_dim = layer_activities[0].shape[1]
        
        # 各層の活動を履歴に追加
        for i, activity in enumerate(layer_activities):
            # 活動の要約統計量を計算（次元削減）
            activity_summary = activity.mean(dim=0).unsqueeze(0)  # [1 x hidden_dim]
            
            # 履歴に追加
            self.cortical_activity_history[i].append(activity_summary)
            
            # 履歴の長さを制限
            if len(self.cortical_activity_history[i]) > self.history_max_length:
                self.cortical_activity_history[i].pop(0)
        
        # 層間の時間差効果を計算
        temporal_integration = torch.zeros(batch_size, hidden_dim, device=layer_activities[0].device)
        
        # シグモイド関数で接続行列を0〜1に正規化
        norm_connections = torch.sigmoid(self.cortical_layer_connections)
        
        # 各層ペアの時間差効果を計算
        for i in range(6):  # 送信層
            for j in range(6):  # 受信層
                if i != j and len(self.cortical_activity_history[i]) > 1:
                    # 送信層からの時間差効果を計算
                    connection_strength = norm_connections[i, j]
                    
                    # 送信層の過去の活動（-2）と現在の活動（-1）の差分を計算
                    if len(self.cortical_activity_history[i]) >= 2:
                        temporal_diff = (self.cortical_activity_history[i][-1] - 
                                        self.cortical_activity_history[i][-2])
                        
                        # 層iから層jへの時間差効果（バッチへの拡張）
                        layer_effect = connection_strength * temporal_diff
                        layer_effect = layer_effect.expand(batch_size, -1)
                        
                        # 時間差効果を累積
                        temporal_integration += layer_effect * self.layer_weights[j]
        
        # 時間差統合を正規化
        if torch.max(torch.abs(temporal_integration)) > 0:
            temporal_integration = temporal_integration / torch.max(torch.abs(temporal_integration))
        
        # 各層の現在の活動を統合（重み付け平均）
        integrated_activity = torch.zeros(batch_size, hidden_dim, device=layer_activities[0].device)
        for i, activity in enumerate(layer_activities):
            integrated_activity += activity * self.layer_weights[i]
        
        # 時間差効果と現在の統合活動を組み合わせる
        final_activity = (1 - self.temporal_importance) * integrated_activity + \
                       self.temporal_importance * temporal_integration
        
        return final_activity
    
    def _update_monitoring(self):
        """
        監視データの更新
        """
        # 実装省略（実際の実装はここに依存）
        pass
    
    def _apply_neuromodulatory_effects(self, h):
        """
        神経伝達物質の効果を適用
        """
        # 実装省略（実際の実装はここに依存）
        return h


def create_biokan_classifier(in_features, hidden_dim=128, num_classes=10, 
                          num_blocks=3, attention_type='biological', 
                          dropout=0.1, neuromodulation=True):
    """
    BioKAN分類器を作成するヘルパー関数
    
    Args:
        in_features: 入力特徴量の数
        hidden_dim: 隠れ層の次元
        num_classes: 出力クラス数
        num_blocks: BioKANブロック数
        attention_type: アテンションの種類
            'biological', 'cortical', 'hierarchical'
        dropout: ドロップアウト確率
        neuromodulation: 神経調節を有効にするかどうか
        
    Returns:
        BioKANモデルのインスタンス
    """
    
    return NeuropharmacologicalBioKAN(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_blocks=num_blocks,
        attention_type=attention_type,
        dropout=dropout
    ) 
    
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
        output = self.classifier(h)
        
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
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # 位置エンコーディング（Transformerの強み）
        self.positional_encoding = self._create_sinusoidal_encoding(max_seq_length, hidden_dim)
        
        # 入力埋め込み層
        self.embedding = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 拡張神経伝達物質システム
        self.neuromodulator = AdvancedNeuromodulatorSystem()
        
        # 神経可塑性モジュール
        self.neuroplasticity = NeuroplasticityModule(hidden_dim) if use_neuroplasticity else None
        
        # 階層的アテンション機構（皮質層ごとに特化）
        self.attention_layers = nn.ModuleList()
        
        # 神経科学的レイヤー名と対応
        self.layer_mapping = {
            0: "layer1",     # 外側皮質層（感覚入力）
            1: "layer2_3",   # 浅層（局所処理）
            2: "layer4",     # 中間層（情報統合）
            3: "layer5",     # 深層（長距離出力）
            4: "layer6",     # 最深層（フィードバック）
            5: "thalamic"    # 視床様レイヤー（調節）
        }
        
        # 各皮質層に特化したアテンション構築
        for i in range(num_layers):
            layer_type = self.layer_mapping.get(i, f"layer{i}")
            
            if layer_type == "layer1":
                # 感覚入力層：高解像度・局所的コンテキスト
                self.attention_layers.append(self._create_layer(
                    hidden_dim, num_heads * 2, dropout, 
                    receptive_field="local", 
                    layer_name=layer_type
                ))
            elif layer_type == "layer2_3":
                # 浅層：水平方向結合による特徴統合
                self.attention_layers.append(self._create_layer(
                    hidden_dim, num_heads, dropout, 
                    receptive_field="horizontal", 
                    layer_name=layer_type
                ))
            elif layer_type == "layer4":
                # 中間層：入力統合・中距離コンテキスト
                self.attention_layers.append(self._create_layer(
                    hidden_dim, num_heads, dropout, 
                    receptive_field="medium", 
                    layer_name=layer_type
                ))
            elif layer_type == "layer5":
                # 深層：広域コンテキスト・長距離依存性
                self.attention_layers.append(self._create_layer(
                    hidden_dim, num_heads // 2, dropout, 
                    receptive_field="global", 
                    layer_name=layer_type
                ))
            elif layer_type == "layer6":
                # 最深層：フィードバック・トップダウン
                self.attention_layers.append(self._create_layer(
                    hidden_dim, num_heads, dropout, 
                    receptive_field="feedback", 
                    layer_name=layer_type
                ))
            else:
                # 視床様レイヤー：調節・ゲーティング
                self.attention_layers.append(self._create_layer(
                    hidden_dim, num_heads, dropout, 
                    receptive_field="gating", 
                    layer_name=layer_type
                ))
        
        # アストロサイトネットワーク（グリア細胞によるレイヤー間調整）
        self.astrocytes = nn.ModuleList([
            EnhancedAstrocyte(region_shape=(hidden_dim // 16, 16))
            for _ in range(num_layers)
        ]) if use_glia else None
        
        # レイヤー間時間差統合
        self.layer_history = [[] for _ in range(num_layers)]
        self.max_history_length = 10
        self.temporal_importance = nn.Parameter(torch.tensor(0.7))
        
        # 視床様ゲーティングメカニズム（注意制御）
        self.thalamic_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 出力分類器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # パラメータ初期化
        self._initialize_parameters()
    
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
    
    def _create_layer(self, hidden_dim, num_heads, dropout, receptive_field, layer_name):
        """
        皮質層に特化したアテンションレイヤーを作成
        """
        if receptive_field == "local":
            # 局所的受容野のアテンション（細かい特徴検出）
            return LocalBiologicalAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                local_context=16,  # 局所コンテキスト窓サイズ
                neuromodulation=True
            )
        elif receptive_field == "horizontal":
            # 水平結合によるアテンション（特徴統合）
            return HorizontalIntegrationAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                neuromodulation=True
            )
        elif receptive_field == "medium":
            # 中距離コンテキストアテンション
            return BiologicalMultiHeadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                neuromodulation=True
            )
        elif receptive_field == "global":
            # グローバルアテンション（長距離依存性）
            return HierarchicalMultiScaleAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_scales=3
            )
        elif receptive_field == "feedback":
            # フィードバックアテンション（トップダウン）
            return FeedbackAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                neuromodulation=True
            )
        else:  # "gating"
            # 視床様ゲーティングアテンション
            return ThalamicAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                neuromodulation=True
            )
    
    def _initialize_parameters(self):
        """
        神経科学的知見に基づくパラメータ初期化
        """
        # 生物学的に妥当な初期化（神経科学的に正しいスケール）
        for name, p in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    # 感覚入力層：正規分布
                    nn.init.normal_(p, mean=0.0, std=0.02)
                elif 'layer1' in name:
                    # 感覚層：弱いシナプス結合
                    nn.init.normal_(p, mean=0.0, std=0.01)
                elif 'layer5' in name:
                    # 出力層：強いシナプス結合
                    nn.init.normal_(p, mean=0.0, std=0.03)
                else:
                    # デフォルト
                    nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, x, mask=None):
        """
        拡張BioKANモデルのフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, in_features] または [batch_size, in_features]
            mask: アテンションマスク（オプション）
            
        Returns:
            出力テンソル [batch_size, num_classes]
        """
        # 入力形状の正規化
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, in_features]
        
        batch_size, seq_len, _ = x.shape
        
        # 入力埋め込み
        h = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # 位置エンコーディングの適用（Transformerの利点）
        if seq_len <= self.max_seq_length:
            h = h + self.pe_scale * self.positional_encoding[:, :seq_len, :].to(h.device)
        
        # 各レイヤーの出力を保存
        layer_outputs = [h]
        
        # レイヤー間の処理
        for i, attn_layer in enumerate(self.attention_layers):
            layer_name = self.layer_mapping.get(i, f"layer{i}")
            
            # 前のレイヤーの状態を保存（時間差処理のため）
            if len(self.layer_history[i]) >= self.max_history_length:
                self.layer_history[i].pop(0)
            self.layer_history[i].append(h.detach())
            
            # 層特有の処理
            residual = h
            
            # アテンション計算
            if layer_name in ["layer1", "layer2_3", "layer4"]:
                # 順伝播経路（フィードフォワード）
                if mask is not None:
                    h, _ = attn_layer(h, h, h, attn_mask=mask)
                else:
                    h, _ = attn_layer(h, h, h)
            
            elif layer_name == "layer5":
                # ワイドコンテキストアテンション（長距離依存性）
                h = attn_layer(h)
            
            elif layer_name == "layer6":
                # フィードバックアテンション（全レイヤーのコンテキスト利用）
                context = torch.cat([out[:, -1:, :] for out in layer_outputs], dim=1)
                h, _ = attn_layer(h, context, context)
            
            else:  # "thalamic"
                # 視床様ゲーティング（選択的注意）
                gate = self.thalamic_gate(h.mean(dim=1, keepdim=True))
                h, _ = attn_layer(h, h, h)
                h = h * gate  # 選択的情報通過
            
            # アストロサイト調節（グリア細胞モジュール）
            if self.astrocytes is not None:
                astrocyte = self.astrocytes[i]
                astro_effects = astrocyte.get_modulatory_effect()
                h = h + astro_effects['trophic_support'] * 0.2
        
        # 最終的な出力
        output = self.classifier(h)
        
        return output

# 不足しているアテンションクラスを実装
class LocalBiologicalAttention(BiologicalMultiHeadAttention):
    """局所的受容野を持つ生物学的アテンション"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, local_context=16, neuromodulation=True):
        super().__init__(embed_dim, num_heads, dropout, neuromodulation)
        self.local_context = local_context
    
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        # 局所的コンテキスト窓の実装
        if key is None:
            key = query
        if value is None:
            value = query
            
        # 局所的マスクの作成
        seq_len = query.size(1)
        local_mask = torch.ones(seq_len, seq_len, device=query.device)
        
        for i in range(seq_len):
            start = max(0, i - self.local_context // 2)
            end = min(seq_len, i + self.local_context // 2 + 1)
            local_mask[i, start:end] = 0
        
        if attn_mask is not None:
            attn_mask = attn_mask + local_mask.bool()
        else:
            attn_mask = local_mask.bool()
        
        return super().forward(query, key, value, attn_mask, need_weights)

# 残りのアテンションクラスを同様に実装
class HorizontalIntegrationAttention(nn.Module):
    """水平結合による特徴統合アテンション
    
    参考文献：
    - Gilbert, C.D., & Wiesel, T.N. (1989). Columnar specificity of intrinsic horizontal and corticocortical connections in cat visual cortex. Journal of Neuroscience, 9(7), 2432-2442.
    - Stettler, D.D., et al. (2002). Lateral connectivity and contextual interactions in macaque primary visual cortex. Neuron, 36(4), 739-750.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, neuromodulation=True):
        super().__init__()
        self.mha = BiologicalMultiHeadAttention(embed_dim, num_heads, dropout, neuromodulation)
        self.lateral_weights = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key=None, value=None, attn_mask=None):
        # 標準的なアテンション
        if key is None:
            key = query
        if value is None:
            value = query
            
        attn_output, attn_weights = self.mha(query, key, value, attn_mask, True)
        
        # 水平方向の結合を追加（V1の水平結合を模倣）
        batch_size, seq_len, dim = query.size()
        lateral_connections = torch.matmul(query, self.lateral_weights)
        
        # 出力の結合と正規化
        output = self.norm(attn_output + 0.5 * lateral_connections)
        
        return output, attn_weights

class FeedbackAttention(BiologicalMultiHeadAttention):
    """フィードバックアテンション（高次皮質から低次皮質へのトップダウン信号）
    
    参考文献：
    - Lamme, V.A., & Roelfsema, P.R. (2000). The distinct modes of vision offered by feedforward and recurrent processing. Trends in Neurosciences, 23(11), 571-579.
    - Gilbert, C.D., & Li, W. (2013). Top-down influences on visual processing. Nature Reviews Neuroscience, 14(5), 350-363.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, neuromodulation=True):
        super().__init__(embed_dim, num_heads, dropout, neuromodulation)
        self.feedback_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.contextual_modulation = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        if key is None:
            key = query
        if value is None:
            value = query
            
        # 通常のアテンションメカニズム
        attn_output, attn_weights = super().forward(query, key, value, attn_mask, True)
        
        # フィードバックゲーティング（トップダウン信号の選択的調整）
        gate = self.feedback_gate(attn_output)
        contextual = self.contextual_modulation(query)
        
        # トップダウン信号による修飾（高次特徴が低次処理を調整）
        modulated_output = attn_output * gate + contextual * (1 - gate)
        
        if need_weights:
            return modulated_output, attn_weights
        return modulated_output

class ThalamicAttention(BiologicalMultiHeadAttention):
    """視床様ゲーティングアテンション
    
    参考文献：
    - Sherman, S.M. (2016). Thalamus plays a central role in ongoing cortical functioning. Nature Neuroscience, 19(4), 533-541.
    - Halassa, M.M., & Kastner, S. (2017). Thalamic functions in distributed cognitive control. Nature Neuroscience, 20(12), 1669-1679.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, neuromodulation=True):
        super().__init__(embed_dim, num_heads, dropout, neuromodulation)
        # 視床網様核のゲーティング機構を模倣
        self.thalamic_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()
        )
        self.arousal_modulation = nn.Parameter(torch.ones(1))
        
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        if key is None:
            key = query
        if value is None:
            value = query
            
        # 入力の情報状態に基づくゲーティング（選択的注意）
        gate = self.thalamic_gate(query)
        gate = gate * self.arousal_modulation  # 覚醒状態による調整
        
        # ゲーティングされた入力で注意メカニズムを適用
        gated_query = query * gate
        
        # 通常のアテンションメカニズム
        attn_output, attn_weights = super().forward(gated_query, key, value, attn_mask, True)
        
        if need_weights:
            return attn_output, attn_weights
        return attn_output
