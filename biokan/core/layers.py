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


class TernaryActivationFunction(Function):
    """
    3価活性化関数 (-1, 0, 1)
    神経系の興奮・抑制・無活性状態を模倣
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(input, torch.tensor(threshold))
        return torch.where(input > threshold, torch.ones_like(input),
                         torch.where(input < -threshold, -torch.ones_like(input),
                                   torch.zeros_like(input)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone() * (input.abs() < threshold).float()
        return grad_input, None


class MultiHeadAttention(nn.Module):
    """
    標準的なマルチヘッドアテンション層
    BioKANの基盤となるアテンション機構
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # QKV投影を計算
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # アテンションスコアの計算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # アテンション適用
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        
        return out


class KANLinear(nn.Module):
    """
    KANモデルの線形層
    入力を3値化して処理する線形変換
    """
    def __init__(self, in_features, out_features, bias=True, threshold=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.threshold = threshold
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        # 3値活性化関数を適用
        ternary_x = TernaryActivationFunction.apply(x, self.threshold)
        output = F.linear(ternary_x, self.weight, self.bias)
        return output 