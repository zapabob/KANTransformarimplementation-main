"""
BioKANモデルの設定管理
"""

from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class NeurotransmitterConfig:
    """神経伝達物質システムの設定"""
    baseline_levels: List[float] = [1.0] * 6
    release_rates: List[float] = [0.1] * 6
    reuptake_rates: List[float] = [0.2] * 6
    decay_rates: List[float] = [0.1] * 6
    synthesis_rates: List[float] = [0.05] * 6

@dataclass
class AttentionConfig:
    """アテンション機構の設定"""
    num_heads: int = 8
    head_dim: int = 64
    dropout: float = 0.1
    use_local_attention: bool = True
    local_window_size: int = 32

@dataclass
class BioKANConfig:
    """BioKANモデルの全体設定"""
    hidden_dim: int = 512
    num_layers: int = 6
    dropout: float = 0.1
    learning_rate: float = 1e-4
    
    # サブシステムの設定
    neurotransmitter: NeurotransmitterConfig = NeurotransmitterConfig()
    attention: AttentionConfig = AttentionConfig()
    
    # 学習設定
    batch_size: int = 32
    num_epochs: int = 100
    
    def to_dict(self) -> Dict:
        """設定を辞書形式に変換"""
        return {
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'neurotransmitter': self.neurotransmitter.__dict__,
            'attention': self.attention.__dict__,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs
        } 