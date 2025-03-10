"""
神経振動を実装するモジュール
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

class NeuralOscillator(nn.Module):
    """
    神経振動子モジュール
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 各周波数帯域のパラメータ
        self.frequencies = nn.Parameter(torch.tensor([
            1.0,   # デルタ波 (1-4 Hz)
            5.0,   # シータ波 (4-8 Hz)
            10.0,  # アルファ波 (8-13 Hz)
            20.0,  # ベータ波 (13-30 Hz)
            40.0   # ガンマ波 (30-100 Hz)
        ]))
        
        # 各振動子の位相
        self.phases = nn.Parameter(torch.rand(5, hidden_dim) * 2 * np.pi)
        
        # 振幅
        self.amplitudes = nn.Parameter(torch.ones(5, hidden_dim) * 0.1)
        
        # 結合強度行列
        self.coupling = nn.Parameter(torch.randn(5, 5) * 0.1)
        
        # 出力への投影
        self.projection = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        
    def kuramoto_model(self, dt: float = 0.001) -> None:
        """
        Kuramotoモデルによる位相の更新
        
        Args:
            dt: 時間ステップ
        """
        # 位相差の計算（ブロードキャストの問題を修正）
        phase_diff = self.phases.unsqueeze(1) - self.phases  # [5, 5, hidden_dim]
        
        # 結合項の計算（einsum操作を修正）
        coupling_term = torch.einsum('ij,ijh->ih', self.coupling, torch.sin(phase_diff))
        
        # 位相の更新
        self.phases.data += dt * (
            self.frequencies.unsqueeze(1) +  # 固有周波数 [5, 1]
            coupling_term  # 結合による影響 [5, hidden_dim]
        )
        
        # 位相を [0, 2π] の範囲に正規化
        self.phases.data = torch.remainder(self.phases.data, 2 * np.pi)
        
    def get_oscillations(self) -> torch.Tensor:
        """
        各振動子の出力を計算
        
        Returns:
            oscillations: 振動出力
        """
        return self.amplitudes * torch.sin(self.phases)  # [5, hidden_dim]
        
    def apply_neuromodulation(self, modulation: Dict[str, torch.Tensor]) -> None:
        """
        神経伝達物質による振動の調節
        
        Args:
            modulation: 神経伝達物質の調節効果
        """
        if 'acetylcholine' in modulation:
            # アセチルコリンによるガンマ波の増強
            self.amplitudes.data[4] *= (1.0 + 0.2 * modulation['acetylcholine'])
            
        if 'serotonin' in modulation:
            # セロトニンによるアルファ波の増強
            self.amplitudes.data[2] *= (1.0 + 0.2 * modulation['serotonin'])
            
        if 'noradrenaline' in modulation:
            # ノルアドレナリンによるベータ波の増強
            self.amplitudes.data[3] *= (1.0 + 0.2 * modulation['noradrenaline'])
            
    def forward(self, x: torch.Tensor, modulation: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            modulation: 神経伝達物質の調節効果
            
        Returns:
            output: 出力テンソル
        """
        # 位相の更新
        self.kuramoto_model()
        
        # 振動の計算
        oscillations = self.get_oscillations()  # [5, hidden_dim]
        
        # 神経伝達物質による調節
        if modulation is not None:
            self.apply_neuromodulation(modulation)
            
        # 入力テンソルと振動成分の結合
        oscillatory_component = torch.matmul(
            oscillations.sum(dim=0),  # [hidden_dim]
            self.projection  # [hidden_dim, hidden_dim]
        )
        
        return x + oscillatory_component
        
class BrainwaveAnalyzer:
    """
    脳波解析クラス
    """
    
    def __init__(self, sampling_rate: float = 1000.0):
        """
        Args:
            sampling_rate: サンプリングレート（Hz）
        """
        self.sampling_rate = sampling_rate
        
        # 周波数帯域の定義
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
    def analyze_power_spectrum(self, signal: torch.Tensor) -> Dict[str, float]:
        """
        パワースペクトル解析
        
        Args:
            signal: 解析する信号
            
        Returns:
            band_powers: 各周波数帯域のパワー
        """
        # FFTの計算
        fft = torch.fft.fft(signal)
        power_spectrum = torch.abs(fft) ** 2
        
        # 周波数軸の生成
        freqs = torch.fft.fftfreq(len(signal), 1.0 / self.sampling_rate)
        
        # 各周波数帯域のパワーを計算
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_powers[band_name] = power_spectrum[mask].mean().item()
            
        return band_powers
        
    def calculate_coherence(
        self,
        signal1: torch.Tensor,
        signal2: torch.Tensor,
        window_size: int = 1000
    ) -> Dict[str, float]:
        """
        コヒーレンス解析
        
        Args:
            signal1: 信号1
            signal2: 信号2
            window_size: 窓サイズ
            
        Returns:
            band_coherence: 各周波数帯域のコヒーレンス
        """
        # 信号を窓に分割
        num_windows = len(signal1) // window_size
        coherence = torch.zeros(window_size // 2 + 1)
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            
            # 各窓でのFFTを計算
            fft1 = torch.fft.rfft(signal1[start:end])
            fft2 = torch.fft.rfft(signal2[start:end])
            
            # クロススペクトルとパワースペクトルの計算
            cross_spectrum = fft1 * torch.conj(fft2)
            power_spectrum1 = torch.abs(fft1) ** 2
            power_spectrum2 = torch.abs(fft2) ** 2
            
            # コヒーレンスの計算
            coherence += torch.abs(cross_spectrum) ** 2 / (power_spectrum1 * power_spectrum2)
            
        # 平均化
        coherence /= num_windows
        
        # 周波数軸の生成
        freqs = torch.fft.rfftfreq(window_size, 1.0 / self.sampling_rate)
        
        # 各周波数帯域のコヒーレンスを計算
        band_coherence = {}
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_coherence[band_name] = coherence[mask].mean().item()
            
        return band_coherence 