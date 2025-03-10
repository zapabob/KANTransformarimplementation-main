"""
神経伝達物質のダイナミクスを実装するモジュール
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

class DetailedNeurotransmitterSystem(nn.Module):
    """
    詳細な神経伝達物質システムの実装
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 神経伝達物質のパラメータ
        self.baseline_levels = nn.Parameter(torch.ones(6))  # [Glu, GABA, DA, 5-HT, NE, ACh]
        self.release_rates = nn.Parameter(torch.ones(6) * 0.1)
        self.reuptake_rates = nn.Parameter(torch.ones(6) * 0.2)
        self.decay_rates = nn.Parameter(torch.ones(6) * 0.1)
        self.synthesis_rates = nn.Parameter(torch.ones(6) * 0.05)
        
        # シナプス小胞プール
        self.vesicle_pools = nn.Parameter(torch.ones(6) * 1.0)
        
        # 現在の神経伝達物質レベル
        self.current_levels = nn.Parameter(torch.ones(6))
        
        # カルシウムダイナミクス
        self.calcium_level = nn.Parameter(torch.tensor(0.1))
        self.calcium_decay = nn.Parameter(torch.tensor(0.1))
        self.calcium_influx_rate = nn.Parameter(torch.tensor(0.2))
        
        # 受容体感受性
        self.receptor_sensitivity = nn.Parameter(torch.ones(6))
        
    def update_neurotransmitter_states(self, activity: torch.Tensor) -> None:
        """
        神経伝達物質の状態を更新
        
        Args:
            activity: 神経活動
        """
        # 活動に基づく放出
        release = torch.sigmoid(activity.mean()) * self.release_rates * self.vesicle_pools
        
        # 再取り込みと分解
        reuptake = self.reuptake_rates * self.current_levels
        decay = self.decay_rates * self.current_levels
        
        # 合成
        synthesis = self.synthesis_rates * (1.0 - self.vesicle_pools)
        
        # 状態の更新
        self.current_levels.data = torch.clamp(
            self.current_levels + release - reuptake - decay,
            0.0, 2.0
        )
        
        self.vesicle_pools.data = torch.clamp(
            self.vesicle_pools - release + synthesis,
            0.0, 2.0
        )
        
    def apply_drug_effects(self, drug_type: str) -> None:
        """
        薬物効果の適用
        
        Args:
            drug_type: 薬物の種類
        """
        if drug_type == "ssri":
            # セロトニン再取り込み阻害
            self.reuptake_rates.data[3] *= 0.3
        elif drug_type == "methylphenidate":
            # ドーパミン・ノルアドレナリン再取り込み阻害
            self.reuptake_rates.data[2:4] *= 0.4
        elif drug_type == "ketamine":
            # NMDA受容体阻害とグルタミン酸調節
            self.receptor_sensitivity.data[0] *= 0.7
        elif drug_type == "psilocybin":
            # セロトニン受容体作動
            self.receptor_sensitivity.data[3] *= 1.5
            
    def update_calcium(self, activity: torch.Tensor) -> None:
        """
        カルシウムダイナミクスの更新
        
        Args:
            activity: 神経活動
        """
        calcium_influx = self.calcium_influx_rate * torch.sigmoid(activity.mean())
        calcium_decay = self.calcium_decay * self.calcium_level
        
        self.calcium_level.data = torch.clamp(
            self.calcium_level + calcium_influx - calcium_decay,
            0.0, 2.0
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            
        Returns:
            modulation: 調節効果
            info: 神経伝達物質の状態情報
        """
        # 神経伝達物質の状態を更新
        self.update_neurotransmitter_states(x)
        self.update_calcium(x)
        
        # 調節効果の計算
        modulation = torch.zeros_like(x)
        
        # グルタミン酸による興奮性調節
        modulation += self.current_levels[0] * self.receptor_sensitivity[0] * x
        
        # GABAによる抑制性調節
        modulation -= self.current_levels[1] * self.receptor_sensitivity[1] * x
        
        # モノアミンによる調節
        for i in range(2, 6):
            modulation *= (1.0 + 0.1 * self.current_levels[i] * self.receptor_sensitivity[i])
            
        # カルシウムの影響
        modulation *= (1.0 + 0.2 * self.calcium_level)
        
        # 情報の収集
        info = {
            'neurotransmitter_levels': self.current_levels.detach(),
            'vesicle_pools': self.vesicle_pools.detach(),
            'calcium_level': self.calcium_level.detach(),
            'receptor_sensitivity': self.receptor_sensitivity.detach()
        }
        
        return modulation, info 