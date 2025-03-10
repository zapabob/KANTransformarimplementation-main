"""
シナプス可塑性を実装するモジュール
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

class SynapticPlasticityModule(nn.Module):
    """
    シナプス可塑性モジュール
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # シナプス重み
        self.weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        
        # STDP（スパイクタイミング依存可塑性）のパラメータ
        self.stdp_window = nn.Parameter(torch.tensor(20.0))  # ミリ秒
        self.stdp_decay = nn.Parameter(torch.tensor(0.1))
        
        # 活動履歴
        self.pre_activity = []
        self.post_activity = []
        
        # ホメオスタティック可塑性のパラメータ
        self.target_activity = nn.Parameter(torch.tensor(0.1))
        self.homeostatic_rate = nn.Parameter(torch.tensor(0.01))
        self.activity_history = torch.zeros(1000)
        self.history_pointer = 0
        
        # メタ可塑性のパラメータ
        self.meta_plasticity = nn.Parameter(torch.ones(hidden_dim))
        self.meta_learning_rate = nn.Parameter(torch.tensor(0.001))
        
    def update_stdp(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        """
        STDPによる重み更新
        
        Args:
            pre: プレシナプス活動 [batch_size, hidden_dim]
            post: ポストシナプス活動 [batch_size, hidden_dim]
        """
        # 活動履歴の更新（バッチ平均を保存）
        self.pre_activity.append(pre.mean(dim=0).detach())  # [hidden_dim]
        self.post_activity.append(post.mean(dim=0).detach())  # [hidden_dim]
        
        if len(self.pre_activity) > 100:
            self.pre_activity.pop(0)
            self.post_activity.pop(0)
        
        # STDPの計算
        for i in range(len(self.pre_activity)):
            dt = (len(self.pre_activity) - i) * self.stdp_window
            
            # 正のタイミング差（pre -> post）
            if i < len(self.pre_activity) - 1:
                positive_dt = torch.exp(-dt * self.stdp_decay)
                # バッチ平均された1次元テンソルで外積を計算
                self.weights.data += 0.01 * positive_dt * torch.outer(
                    self.post_activity[-1],  # [hidden_dim]
                    self.pre_activity[i]     # [hidden_dim]
                )
            
            # 負のタイミング差（post -> pre）
            if i > 0:
                negative_dt = torch.exp(-dt * self.stdp_decay)
                # バッチ平均された1次元テンソルで外積を計算
                self.weights.data -= 0.01 * negative_dt * torch.outer(
                    self.post_activity[i],   # [hidden_dim]
                    self.pre_activity[-1]    # [hidden_dim]
                )
                
        # 重みの範囲を制限
        self.weights.data = torch.clamp(self.weights, -1.0, 1.0)
        
    def update_homeostatic(self, activity: torch.Tensor) -> None:
        """
        ホメオスタティック可塑性の更新
        
        Args:
            activity: 神経活動
        """
        # 活動履歴の更新
        self.activity_history[self.history_pointer] = activity.mean().item()
        self.history_pointer = (self.history_pointer + 1) % 1000
        
        # 平均活動の計算
        mean_activity = self.activity_history.mean()
        
        # 活動差に基づく調整
        activity_diff = self.target_activity - mean_activity
        self.weights.data *= (1.0 + self.homeostatic_rate * activity_diff)
        
        # 重みの範囲を制限
        self.weights.data = torch.clamp(self.weights, -1.0, 1.0)
        
    def update_meta_plasticity(self, activity: torch.Tensor) -> None:
        """
        メタ可塑性の更新
        
        Args:
            activity: 神経活動
        """
        # 活動レベルに基づくメタ可塑性の調整
        activity_level = activity.mean(dim=0)
        self.meta_plasticity.data += self.meta_learning_rate * (
            activity_level - self.meta_plasticity
        )
        
        # メタ可塑性の範囲を制限
        self.meta_plasticity.data = torch.clamp(self.meta_plasticity, 0.0, 2.0)
        
    def apply_drug_modulation(self, modulation: Dict[str, torch.Tensor]) -> None:
        """
        神経伝達物質による可塑性の調節
        
        Args:
            modulation: 神経伝達物質の調節効果
        """
        if 'plasticity' in modulation:
            self.weights.data *= (1.0 + 0.1 * modulation['plasticity'])
            
        if 'calcium' in modulation:
            calcium_effect = torch.sigmoid(modulation['calcium'])
            self.weights.data *= (1.0 + 0.2 * calcium_effect)
            
    def forward(self, x: torch.Tensor, modulation: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            modulation: 神経伝達物質の調節効果
            
        Returns:
            output: 出力テンソル
        """
        # シナプス伝達
        output = torch.matmul(x, self.weights * self.meta_plasticity)
        
        # 可塑性の更新
        self.update_stdp(x, output)
        self.update_homeostatic(output)
        self.update_meta_plasticity(output)
        
        # 神経伝達物質による調節
        if modulation is not None:
            self.apply_drug_modulation(modulation)
            
        return output 