"""
神経伝達物質のダイナミクスと調節機構の実装
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

class DetailedNeurotransmitterSystem(nn.Module):
    """
    詳細な神経伝達物質のダイナミクスを実装するクラス
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # 神経伝達物質の初期パラメータ
        self.neurotransmitters = {
            'glutamate': {
                'baseline': 1.0,
                'release_rate': 0.8,
                'reuptake_rate': 0.6,
                'decay_rate': 0.2,
                'synthesis_rate': 0.4,
                'vesicle_pool': 1.0,
                'current_level': 1.0
            },
            'gaba': {
                'baseline': 1.0,
                'release_rate': 0.7,
                'reuptake_rate': 0.5,
                'decay_rate': 0.15,
                'synthesis_rate': 0.3,
                'vesicle_pool': 1.0,
                'current_level': 1.0
            },
            'dopamine': {
                'baseline': 1.0,
                'release_rate': 0.6,
                'reuptake_rate': 0.7,
                'decay_rate': 0.1,
                'synthesis_rate': 0.2,
                'vesicle_pool': 1.0,
                'current_level': 1.0
            },
            'serotonin': {
                'baseline': 1.0,
                'release_rate': 0.5,
                'reuptake_rate': 0.8,
                'decay_rate': 0.05,
                'synthesis_rate': 0.15,
                'vesicle_pool': 1.0,
                'current_level': 1.0
            },
            'noradrenaline': {
                'baseline': 1.0,
                'release_rate': 0.7,
                'reuptake_rate': 0.6,
                'decay_rate': 0.1,
                'synthesis_rate': 0.25,
                'vesicle_pool': 1.0,
                'current_level': 1.0
            },
            'acetylcholine': {
                'baseline': 1.0,
                'release_rate': 0.6,
                'reuptake_rate': 0.5,
                'decay_rate': 0.15,
                'synthesis_rate': 0.3,
                'vesicle_pool': 1.0,
                'current_level': 1.0
            }
        }
        
        # シナプス前終末の状態を表現
        self.presynaptic_state = nn.Parameter(torch.ones(hidden_dim))
        
        # 受容体の感受性
        self.receptor_sensitivity = nn.Parameter(torch.ones(len(self.neurotransmitters)))
        
        # カルシウムダイナミクス
        self.calcium_level = nn.Parameter(torch.ones(hidden_dim))
        self.calcium_channels = nn.Parameter(torch.rand(hidden_dim) * 0.1)
        
        # 薬物効果の状態
        self.active_drugs = {}
        
    def update_neurotransmitter(self, nt_name: str, activity: float, delta_t: float = 0.1) -> None:
        """
        神経伝達物質の状態を更新
        
        Args:
            nt_name: 神経伝達物質の名前
            activity: 神経活動レベル
            delta_t: 時間ステップ
        """
        nt = self.neurotransmitters[nt_name]
        
        # シナプス小胞からの放出
        release = activity * nt['release_rate'] * nt['vesicle_pool'] * delta_t
        
        # 再取り込みと分解
        reuptake = nt['current_level'] * nt['reuptake_rate'] * delta_t
        decay = nt['current_level'] * nt['decay_rate'] * delta_t
        
        # 合成
        synthesis = (nt['baseline'] - nt['vesicle_pool']) * nt['synthesis_rate'] * delta_t
        
        # 状態の更新
        nt['current_level'] += release - reuptake - decay
        nt['vesicle_pool'] += synthesis - release
        
        # 値の範囲を制限
        nt['current_level'] = max(0.0, min(2.0, nt['current_level']))
        nt['vesicle_pool'] = max(0.0, min(2.0, nt['vesicle_pool']))
        
    def apply_drug_effect(self, drug_name: str, dose: float, duration: float) -> None:
        """
        薬物効果を適用
        
        Args:
            drug_name: 薬物の名前
            dose: 投与量
            duration: 効果持続時間
        """
        drug_effects = {
            'ssri': {
                'serotonin': {'reuptake_rate': -0.5},
            },
            'methylphenidate': {
                'dopamine': {'reuptake_rate': -0.4},
                'noradrenaline': {'reuptake_rate': -0.3}
            },
            'ketamine': {
                'glutamate': {'release_rate': -0.3},
                'dopamine': {'release_rate': 0.2}
            },
            'psilocybin': {
                'serotonin': {'receptor_sensitivity': 0.4},
                'glutamate': {'release_rate': 0.2}
            }
        }
        
        if drug_name in drug_effects:
            self.active_drugs[drug_name] = {
                'effects': drug_effects[drug_name],
                'dose': dose,
                'duration': duration,
                'elapsed_time': 0.0
            }
            
    def update_calcium_dynamics(self, activity: torch.Tensor, delta_t: float = 0.1) -> None:
        """
        カルシウムダイナミクスの更新
        
        Args:
            activity: 神経活動
            delta_t: 時間ステップ
        """
        # カルシウムチャネルの活性化
        channel_activation = torch.sigmoid(activity)
        
        # カルシウム流入
        calcium_influx = channel_activation * self.calcium_channels
        
        # カルシウム排出（指数減衰）
        calcium_efflux = self.calcium_level * 0.1
        
        # カルシウムレベルの更新
        self.calcium_level.data += (calcium_influx - calcium_efflux) * delta_t
        
        # 値の範囲を制限
        self.calcium_level.data = torch.clamp(self.calcium_level, 0.0, 2.0)
        
    def forward(self, activity: torch.Tensor, delta_t: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        神経伝達物質システムの順伝播
        
        Args:
            activity: 神経活動
            delta_t: 時間ステップ
            
        Returns:
            modulation: 調節効果の辞書
        """
        # カルシウムダイナミクスの更新
        self.update_calcium_dynamics(activity, delta_t)
        
        # 各神経伝達物質の更新
        for nt_name in self.neurotransmitters:
            self.update_neurotransmitter(nt_name, activity.mean().item(), delta_t)
            
        # 薬物効果の適用
        for drug_name, drug_info in self.active_drugs.items():
            drug_info['elapsed_time'] += delta_t
            if drug_info['elapsed_time'] > drug_info['duration']:
                del self.active_drugs[drug_name]
                continue
                
            for nt_name, effects in drug_info['effects'].items():
                for param, change in effects.items():
                    if param in self.neurotransmitters[nt_name]:
                        self.neurotransmitters[nt_name][param] += change * drug_info['dose']
        
        # 調節効果の計算
        modulation = {
            'excitation': torch.sigmoid(
                self.neurotransmitters['glutamate']['current_level'] * self.receptor_sensitivity[0] +
                self.neurotransmitters['noradrenaline']['current_level'] * self.receptor_sensitivity[4]
            ),
            'inhibition': torch.sigmoid(
                self.neurotransmitters['gaba']['current_level'] * self.receptor_sensitivity[1]
            ),
            'plasticity': torch.sigmoid(
                self.neurotransmitters['dopamine']['current_level'] * self.receptor_sensitivity[2] +
                self.neurotransmitters['serotonin']['current_level'] * self.receptor_sensitivity[3]
            ),
            'attention': torch.sigmoid(
                self.neurotransmitters['acetylcholine']['current_level'] * self.receptor_sensitivity[5]
            ),
            'calcium': self.calcium_level
        }
        
        return modulation 