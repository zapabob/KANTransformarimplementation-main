"""
神経調節に関連するユーティリティ関数
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

def calculate_neurotransmitter_levels(
    baseline_levels: List[float],
    synthesis_rates: List[float],
    decay_rates: List[float],
    current_time: float,
    last_spike_times: Optional[List[float]] = None
) -> torch.Tensor:
    """神経伝達物質レベルを計算する

    Args:
        baseline_levels: 各神経伝達物質の基準レベル
        synthesis_rates: 各神経伝達物質の合成率
        decay_rates: 各神経伝達物質の減衰率
        current_time: 現在の時刻
        last_spike_times: 最後のスパイク発生時刻（オプション）

    Returns:
        各神経伝達物質の現在のレベル
    """
    levels = torch.tensor(baseline_levels, dtype=torch.float32)
    
    if last_spike_times is not None:
        for i, (synthesis_rate, decay_rate, last_spike) in enumerate(
            zip(synthesis_rates, decay_rates, last_spike_times)
        ):
            if last_spike is not None:
                time_since_spike = current_time - last_spike
                # 合成と減衰を考慮してレベルを更新
                delta = synthesis_rate * np.exp(-decay_rate * time_since_spike)
                levels[i] += delta
    
    return levels

def apply_neuromodulation(
    activations: torch.Tensor,
    neurotransmitter_levels: torch.Tensor,
    modulation_weights: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """活性化に神経調節を適用する

    Args:
        activations: ニューロンの活性化
        neurotransmitter_levels: 各神経伝達物質のレベル
        modulation_weights: 神経伝達物質ごとの調節重み

    Returns:
        調節された活性化
    """
    modulated = activations.clone()
    
    for nt_idx, (nt_name, weights) in enumerate(modulation_weights.items()):
        nt_level = neurotransmitter_levels[nt_idx]
        modulation = weights * nt_level
        modulated *= (1 + modulation)
    
    return modulated

def update_spike_times(
    activations: torch.Tensor,
    threshold: float,
    current_time: float,
    last_spike_times: List[Optional[float]]
) -> List[Optional[float]]:
    """スパイク発生時刻を更新する

    Args:
        activations: ニューロンの活性化
        threshold: スパイク発生の閾値
        current_time: 現在の時刻
        last_spike_times: 最後のスパイク発生時刻のリスト

    Returns:
        更新されたスパイク発生時刻のリスト
    """
    spikes = activations > threshold
    new_spike_times = last_spike_times.copy()
    
    for i, spike in enumerate(spikes):
        if spike:
            new_spike_times[i] = current_time
    
    return new_spike_times

def calculate_diffusion(
    neurotransmitter_levels: torch.Tensor,
    diffusion_rates: List[float],
    connectivity_matrix: torch.Tensor
) -> torch.Tensor:
    """神経伝達物質の拡散を計算する

    Args:
        neurotransmitter_levels: 各神経伝達物質の現在のレベル
        diffusion_rates: 各神経伝達物質の拡散率
        connectivity_matrix: ニューロン間の接続行列

    Returns:
        拡散後の神経伝達物質レベル
    """
    diffused_levels = neurotransmitter_levels.clone()
    
    for i, diff_rate in enumerate(diffusion_rates):
        # 拡散項の計算
        diffusion = diff_rate * (
            torch.matmul(connectivity_matrix, neurotransmitter_levels[i]) -
            neurotransmitter_levels[i]
        )
        diffused_levels[i] += diffusion
    
    return torch.clamp(diffused_levels, min=0.0)  # 負の値を防ぐ 