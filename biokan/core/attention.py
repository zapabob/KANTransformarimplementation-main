import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

class BiologicalMultiHeadAttention(nn.Module):
    """
    生物学的特性を模倣したマルチヘッドアテンション
    - 注意の選択性と持続性を模倣
    - 神経調節による注意調整メカニズム
    - 階層的時間スケールでの情報処理
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        bias: bool = True,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        neuromodulated: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.neuromodulated = neuromodulated
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # 通常のキー・クエリ・バリュー線形投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 時間スケール調整パラメータ（デフォルトは標準Transformerと同等）
        self.time_scales = nn.Parameter(torch.ones(num_heads))
        
        # 注意持続性パラメータ（アテンションの「記憶」）
        self.attention_memory = None
        self.memory_decay = nn.Parameter(torch.tensor(0.9))
        
        # ゼロアテンション拡張機能
        self.add_zero_attn = add_zero_attn
        
        # 神経調節用の調整機構（neuromodulated=Trueの場合にのみ使用）
        if neuromodulated:
            # 神経調節による注意調整
            self.neuromod_gates = nn.Linear(embed_dim, num_heads)
            self.neuromod_bias = nn.Parameter(torch.zeros(num_heads))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """パラメータの初期化"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        
        # 時間スケールを正の値で初期化
        with torch.no_grad():
            self.time_scales.data = torch.clamp(self.time_scales.data, min=0.1, max=10.0)
    
    def apply_neuromodulation(
        self, 
        attn_weights: torch.Tensor, 
        neuromod_state: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        神経調節による注意調整
        
        Args:
            attn_weights: アテンション重み [batch_size, num_heads, seq_len, seq_len]
            neuromod_state: 神経調節状態の辞書
            
        Returns:
            調整されたアテンション重み
        """
        if not self.neuromodulated or neuromod_state is None:
            return attn_weights
        
        batch_size = attn_weights.size(0)
        
        # 神経調節効果の計算
        effects = {
            # 注意の選択性を調整（ノルアドレナリン効果）
            'noradrenaline': neuromod_state.get('noradrenaline', 0.0),
            # 注意の持続性に影響（アセチルコリン効果）
            'acetylcholine': neuromod_state.get('acetylcholine', 0.0),
            # 注意の切り替えに影響（ドーパミン効果）
            'dopamine': neuromod_state.get('dopamine', 0.0),
            # 情動による注意バイアス（セロトニン効果）
            'serotonin': neuromod_state.get('serotonin', 0.0)
        }
        
        # 効果の適用
        # 1. ノルアドレナリン: 注意の選択性（コントラスト）を調整
        if abs(effects['noradrenaline']) > 0.05:
            temperature = 1.0 / (1.0 + effects['noradrenaline'] * 0.5)
            attn_weights = attn_weights / temperature
        
        # 2. アセチルコリン: 注意の持続性と詳細度
        if effects['acetylcholine'] > 0.1:
            # 詳細な特徴への注意を強化（対角成分を強調）
            diag_mask = torch.eye(attn_weights.size(-1), device=attn_weights.device)
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
            attn_weights = attn_weights * (1.0 + effects['acetylcholine'] * 0.3 * diag_mask)
        
        # 3. ドーパミン: 注意のシフトと探索
        if abs(effects['dopamine']) > 0.1:
            # 正のドーパミン: より広範な探索（アテンションをより均一に）
            if effects['dopamine'] > 0:
                noise = torch.randn_like(attn_weights) * effects['dopamine'] * 0.1
                attn_weights = attn_weights + noise
            # 負のドーパミン: より局所的な注意（トップkの注意をさらに強化）
            else:
                neg_effect = -effects['dopamine']
                topk_values, _ = torch.topk(attn_weights, k=max(1, int(attn_weights.size(-1) * 0.2)), dim=-1)
                min_topk = topk_values[:, :, :, -1].unsqueeze(-1)
                mask = (attn_weights >= min_topk).float()
                attn_weights = attn_weights * (1.0 + neg_effect * 0.3 * mask)
        
        # 4. セロトニン: 情動バイアス（過去の経験に基づく注意）
        if self.attention_memory is not None and abs(effects['serotonin']) > 0.05:
            memory_influence = torch.sigmoid(torch.tensor(effects['serotonin'] * 2.0))
            attn_weights = (1.0 - memory_influence) * attn_weights + memory_influence * self.attention_memory
        
        return attn_weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        neuromod_state: Optional[Dict[str, float]] = None,
        average_attn_weights: bool = True,
        return_attn_patterns: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        マルチヘッドアテンションの順伝播
        
        Args:
            query: クエリテンソル [batch_size, seq_len, embed_dim] or [seq_len, batch_size, embed_dim]
            key: キーテンソル [batch_size, seq_len, embed_dim] or [seq_len, batch_size, embed_dim]
            value: バリューテンソル [batch_size, seq_len, embed_dim] or [seq_len, batch_size, embed_dim]
            key_padding_mask: パディングマスク [batch_size, seq_len]
            need_weights: アテンション重みを返すかどうか
            attn_mask: アテンションマスク [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            neuromod_state: 神経調節状態の辞書
            average_attn_weights: ヘッド間でアテンション重みを平均化するかどうか
            return_attn_patterns: すべてのアテンションパターンを返すかどうか
            
        Returns:
            出力テンソル [batch_size, seq_len, embed_dim] or [seq_len, batch_size, embed_dim]
            アテンション重み [batch_size, seq_len, seq_len] or [batch_size, num_heads, seq_len, seq_len]
        """
        # バッチファーストに変換
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # クエリ、キー、バリューの投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 各ヘッドに分割
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim)
        
        # トランスポーズして形状を調整: [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # ゼロアテンションの追加（オプション）
        if self.add_zero_attn:
            src_len += 1
            zero_attn = torch.zeros((batch_size, self.num_heads, 1, self.head_dim), dtype=k.dtype, device=k.device)
            k = torch.cat([k, zero_attn], dim=2)
            v = torch.cat([v, zero_attn], dim=2)
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1), value=1)
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = F.pad(attn_mask, (0, 1), value=0)
                else:
                    attn_mask = F.pad(attn_mask, (0, 1), value=0)
        
        # 時間スケール調整: 各ヘッドの時間スケールを調整
        time_scale_factors = self.time_scales.view(1, self.num_heads, 1, 1)
        k = k * time_scale_factors
        
        # スケーリングドットプロダクトアテンション
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # マスキングの適用
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_weights += attn_mask
            else:
                attn_weights += attn_mask
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # 神経調節の適用
        if self.neuromodulated and neuromod_state is not None:
            attn_weights = self.apply_neuromodulation(attn_weights, neuromod_state)
        
        # ソフトマックスでアテンション重みを正規化
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # アテンションメモリの更新
        if self.training:
            if self.attention_memory is None:
                self.attention_memory = attn_weights.detach()
            else:
                # 指数移動平均でメモリを更新
                decay = self.memory_decay.sigmoid().item()
                self.attention_memory = (decay * self.attention_memory + 
                                       (1.0 - decay) * attn_weights.detach())
        
        # アテンション適用
        attn_output = torch.matmul(attn_weights, v)
        
        # 形状を元に戻す
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        
        # 出力投影
        attn_output = self.out_proj(attn_output)
        
        # 必要に応じてバッチファーストを元に戻す
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            
            if not self.batch_first:
                # 戻り値の形状: [tgt_len, src_len] for compatibility
                attn_weights = attn_weights.transpose(0, 1)
            
            return attn_output, attn_weights
        else:
            if return_attn_patterns:
                return attn_output, attn_weights
            else:
                return attn_output, None


class HierarchicalMultiScaleAttention(nn.Module):
    """
    階層的マルチスケールアテンション
    - 異なる時間スケールや空間スケールでの情報処理
    - 大脳皮質の階層的処理を模倣
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8,
        num_scales: int = 3,
        dropout: float = 0.1,
        bias: bool = True,
        add_zero_attn: bool = False,
        neuromodulated: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.neuromodulated = neuromodulated
        
        # 各スケールのヘッド数を計算
        heads_per_scale = [max(1, num_heads // (2 ** i)) for i in range(num_scales)]
        self.heads_per_scale = heads_per_scale
        
        # 各スケールのアテンションモジュール
        self.attention_modules = nn.ModuleList([
            BiologicalMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=heads,
                dropout=dropout,
                bias=bias,
                add_zero_attn=add_zero_attn,
                batch_first=True,
                neuromodulated=neuromodulated
            )
            for heads in heads_per_scale
        ])
        
        # スケール間の重み付け
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 出力投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # スケール選択ゲート（神経調節による制御）
        if neuromodulated:
            self.scale_gate = nn.Linear(embed_dim, num_scales)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        neuromod_state: Optional[Dict[str, float]] = None,
        return_attn_patterns: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        階層的マルチスケールアテンションの順伝播
        
        Args:
            query: クエリテンソル [batch_size, seq_len, embed_dim]
            key: キーテンソル [batch_size, seq_len, embed_dim]
            value: バリューテンソル [batch_size, seq_len, embed_dim]
            key_padding_mask: パディングマスク [batch_size, seq_len]
            attn_mask: アテンションマスク [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            neuromod_state: 神経調節状態の辞書
            return_attn_patterns: アテンションパターンを返すかどうか
            
        Returns:
            出力テンソル [batch_size, seq_len, embed_dim]
            アテンションパターンのリスト（オプション）
        """
        batch_size, seq_len, _ = query.shape
        
        # 神経調節による各スケールの重み付け
        scale_weights = self.scale_weights
        if self.neuromodulated and neuromod_state is not None:
            # ドーパミン: 探索と柔軟性のバランス（高→細かいスケールを重視）
            dopamine = neuromod_state.get('dopamine', 0.0)
            # ノルアドレナリン: 注意の広さ（高→広いスケールを重視）
            noradrenaline = neuromod_state.get('noradrenaline', 0.0)
            
            # 神経調節効果
            scale_bias = torch.zeros_like(scale_weights)
            for i in range(self.num_scales):
                # スケールに対する特定の効果を計算
                if i == 0:  # 最も細かいスケール
                    scale_bias[i] += dopamine * 0.2
                elif i == self.num_scales - 1:  # 最も粗いスケール
                    scale_bias[i] += noradrenaline * 0.2
            
            # スケール重みを調整
            scale_weights = F.softmax(self.scale_weights + scale_bias, dim=0)
        
        # 各スケールでのアテンション計算
        outputs = []
        attn_patterns = []
        
        for i, attn_module in enumerate(self.attention_modules):
            # スケールに応じたアテンションマスク調整（オプション）
            current_attn_mask = attn_mask
            
            # より粗いスケールでは、近傍トークンへのアテンションを制限
            if i > 0 and attn_mask is None:
                # 各スケールでの注意幅
                attention_span = min(seq_len, 2 ** (i + 1))
                
                # 帯状マスクを作成（対角線から±attention_span/2の範囲のみアテンション可能）
                mask = torch.ones(seq_len, seq_len, device=query.device)
                for j in range(seq_len):
                    half_span = attention_span // 2
                    start = max(0, j - half_span)
                    end = min(seq_len, j + half_span + 1)
                    mask[j, start:end] = 0
                
                # マスク変換（0→許可、float('-inf')→禁止）
                current_attn_mask = mask * float('-inf')
            
            # スケール別アテンション計算
            output, attn = attn_module(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=return_attn_patterns,
                attn_mask=current_attn_mask,
                neuromod_state=neuromod_state,
                average_attn_weights=False,
                return_attn_patterns=True
            )
            
            outputs.append(output)
            if return_attn_patterns and attn is not None:
                attn_patterns.append(attn)
        
        # 各スケールの出力を重み付け和
        combined_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            combined_output += scale_weights[i] * output
        
        # 最終出力投影
        final_output = self.out_proj(combined_output)
        
        if return_attn_patterns:
            return final_output, attn_patterns
        else:
            return final_output, None


class CorticalAttention(nn.Module):
    """
    大脳皮質の層構造を模倣したアテンション機構
    - 各皮質層（層1〜6）の特性に基づいたマルチアテンション
    - フィードフォワードとフィードバック経路の統合
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        neuromodulated: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.neuromodulated = neuromodulated
        
        # 大脳皮質の主要層（L2/3, L4, L5, L6）を模倣した4種類のアテンション
        # L2/3: 皮質間結合（水平統合）
        self.L23_attention = BiologicalMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            neuromodulated=neuromodulated
        )
        
        # L4: 入力層（視床からの入力受信）
        self.L4_attention = BiologicalMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads // 2,
            dropout=dropout,
            bias=bias,
            neuromodulated=neuromodulated
        )
        
        # L5: 出力層（他の皮質領域や皮質下領域への投射）
        self.L5_attention = HierarchicalMultiScaleAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_scales=2,
            dropout=dropout,
            bias=bias,
            neuromodulated=neuromodulated
        )
        
        # L6: フィードバック制御層
        self.L6_attention = BiologicalMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads // 2,
            dropout=dropout,
            bias=bias,
            neuromodulated=neuromodulated
        )
        
        # 層間の結合と統合
        self.L4_to_L23 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.L23_to_L5 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.L5_to_L6 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.L6_to_L4 = nn.Linear(embed_dim, embed_dim, bias=bias)  # フィードバック
        
        # 最終出力投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 神経調節ゲート（各層の活性調整）
        if neuromodulated:
            self.neuromod_gates = nn.Linear(embed_dim, 4)  # 4層の活性ゲート
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        neuromod_state: Optional[Dict[str, float]] = None,
        return_attn_patterns: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        大脳皮質型アテンションの順伝播
        
        Args:
            query: クエリテンソル [batch_size, seq_len, embed_dim]
            key: キーテンソル [batch_size, seq_len, embed_dim]
            value: バリューテンソル [batch_size, seq_len, embed_dim]
            key_padding_mask: パディングマスク [batch_size, seq_len]
            attn_mask: アテンションマスク
            neuromod_state: 神経調節状態の辞書
            return_attn_patterns: アテンションパターンを返すかどうか
            
        Returns:
            出力テンソル [batch_size, seq_len, embed_dim]
            アテンションパターンの辞書（オプション）
        """
        attn_patterns = {} if return_attn_patterns else None
        
        # 神経調節ゲートの計算
        layer_gates = torch.ones(4, device=query.device)
        if self.neuromodulated and neuromod_state is not None:
            # 特定の神経伝達物質の効果
            acetylcholine = neuromod_state.get('acetylcholine', 0.0)  # 詳細処理
            noradrenaline = neuromod_state.get('noradrenaline', 0.0)  # 警戒・注意
            dopamine = neuromod_state.get('dopamine', 0.0)  # 目標指向
            serotonin = neuromod_state.get('serotonin', 0.0)  # 情動調整
            
            # 神経伝達物質による層活性の調整
            layer_gates = torch.tensor([
                1.0 + 0.2 * acetylcholine,  # L2/3
                1.0 + 0.2 * noradrenaline,  # L4
                1.0 + 0.2 * dopamine,       # L5
                1.0 + 0.2 * serotonin       # L6
            ], device=query.device)
            
            # 活性値を0.5〜1.5の範囲に制限
            layer_gates = torch.clamp(layer_gates, min=0.5, max=1.5)
        
        # Layer 4: 入力層（入力の初期処理）
        L4_out, L4_attn = self.L4_attention(
            query, key, value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            neuromod_state=neuromod_state,
            return_attn_patterns=return_attn_patterns
        )
        L4_out = L4_out * layer_gates[1]
        if return_attn_patterns:
            attn_patterns['L4'] = L4_attn
        
        # Layer 2/3: 皮質間結合層（水平統合）
        # L4からの入力を受け取る
        L23_input = self.L4_to_L23(L4_out)
        L23_out, L23_attn = self.L23_attention(
            L23_input, L23_input, L23_input,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            neuromod_state=neuromod_state,
            return_attn_patterns=return_attn_patterns
        )
        L23_out = L23_out * layer_gates[0]
        if return_attn_patterns:
            attn_patterns['L23'] = L23_attn
        
        # Layer 5: 出力層（階層的処理）
        # L2/3からの入力を受け取る
        L5_input = self.L23_to_L5(L23_out)
        L5_out, L5_attn = self.L5_attention(
            L5_input, L5_input, L5_input,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            neuromod_state=neuromod_state,
            return_attn_patterns=return_attn_patterns
        )
        L5_out = L5_out * layer_gates[2]
        if return_attn_patterns:
            attn_patterns['L5'] = L5_attn
        
        # Layer 6: フィードバック制御層
        # L5からの入力を受け取る
        L6_input = self.L5_to_L6(L5_out)
        L6_out, L6_attn = self.L6_attention(
            L6_input, key, value,  # クエリはL6、キー・バリューは元の入力
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            neuromod_state=neuromod_state,
            return_attn_patterns=return_attn_patterns
        )
        L6_out = L6_out * layer_gates[3]
        if return_attn_patterns:
            attn_patterns['L6'] = L6_attn
        
        # フィードバック信号（L6からL4へ）
        feedback = self.L6_to_L4(L6_out)
        
        # 最終出力は主にL5の出力を使用し、L6からのフィードバックを加味
        output = L5_out + 0.2 * feedback
        
        # 最終出力投影
        output = self.out_proj(output)
        
        return output, attn_patterns if return_attn_patterns else None 