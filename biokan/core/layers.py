"""
BioKAN 基本レイヤーモジュール
3価活性化関数などの基本的なニューラルネットワークレイヤーを提供
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from typing import Dict, List, Optional, Tuple, Any, Union


class TernaryActivationFunction(nn.Module):
    """
    3価活性化関数 (-1, 0, 1)
    神経系の興奮・抑制・無活性状態を模倣
    """
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, x):
        return torch.where(x > self.threshold, torch.ones_like(x),
                         torch.where(x < -self.threshold, -torch.ones_like(x),
                                   torch.zeros_like(x)))


class MultiHeadAttention(nn.Module):
    """
    標準的なマルチヘッドアテンション層
    BioKANの基盤となるアテンション機構
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        if key is None:
            key = query
        if value is None:
            value = key
            
        batch_size = query.size(0)
        
        # 線形変換とヘッドの分割
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # スケーリングされたドット積アテンション
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 値との積と結合
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        output = self.out_proj(attn_output)
        
        if need_weights:
            return output, attn_weights
        return output, None


class KANLinear(nn.Linear):
    """
    KANモデルの線形層
    入力を3値化して処理する線形変換
    """
    def __init__(self, in_features, out_features, bias=True, neuromodulation=True):
        super().__init__(in_features, out_features, bias)
        self.neuromodulation = neuromodulation
        if neuromodulation:
            self.neuromod_scale = nn.Parameter(torch.ones(out_features))
            self.neuromod_bias = nn.Parameter(torch.zeros(out_features))
            
    def forward(self, x, neuromod=None):
        output = super().forward(x)
        if self.neuromodulation and neuromod is not None:
            output = output * self.neuromod_scale + self.neuromod_bias
        return output 