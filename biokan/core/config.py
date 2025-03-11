"""
BioKANモデルの設定クラス
"""

from dataclasses import dataclass, field
from typing import List, Dict
from typing import Optional
import torch


def default_baseline_levels() -> List[float]:
    return [0.5, 0.5, 0.5, 0.5]


def default_neurotransmitter_config() -> Dict[str, float]:
    return {
        "dopamine": 0.5,
        "serotonin": 0.5,
        "noradrenaline": 0.5,
        "acetylcholine": 0.5
    }


def default_synthesis_rates() -> List[float]:
    return [0.05] * 6


def default_decay_rates() -> List[float]:
    return [0.01] * 6


def default_diffusion_rates() -> List[float]:
    return [0.03] * 6


@dataclass
class NeurotransmitterConfig:
    """神経伝達物質の設定"""
    baseline_levels: List[float] = field(default_factory=default_baseline_levels)
    synthesis_rates: List[float] = field(default_factory=default_synthesis_rates)
    decay_rates: List[float] = field(default_factory=default_decay_rates)
    diffusion_rates: List[float] = field(default_factory=default_diffusion_rates)


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
    """BioKANモデルの設定を保持するクラス"""
    
    # モデルのアーキテクチャ
    input_dim: int = 784  # MNIST用
    hidden_dim: int = 512
    output_dim: int = 10  # MNIST用
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    attention_type: str = "biological"
    use_neuromodulation: bool = True
    use_batch_norm: bool = True  # バッチ正規化の使用
    
    # 訓練パラメータ
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    
    # その他の設定
    device: str = "cuda"
    
    def __post_init__(self):
        """初期化後の処理
        hidden_dimがnum_headsで割り切れるように調整
        """
        if self.hidden_dim % self.num_heads != 0:
            self.hidden_dim = ((self.hidden_dim + self.num_heads - 1) 
                             // self.num_heads) * self.num_heads
            print(f"hidden_dimを{self.hidden_dim}に調整しました（num_headsで割り切れるように）")
        
        """設定値の検証"""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dimは正の整数である必要があります")
        
        if self.num_layers <= 0:
            raise ValueError("num_layersは正の整数である必要があります")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropoutは0から1の間である必要があります")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rateは正の値である必要があります")
        
        if self.batch_size <= 0:
            raise ValueError("batch_sizeは正の整数である必要があります")
        
        if self.num_epochs <= 0:
            raise ValueError("num_epochsは正の整数である必要があります")
        
        valid_attention_types = ["biological", "cortical", "hierarchical"]
        if self.attention_type not in valid_attention_types:
            raise ValueError(
                f"attention_typeは{valid_attention_types}のいずれかである必要があります"
            )

    def to_dict(self) -> Dict:
        """設定をディクショナリ形式で返す"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'device': self.device,
            'use_neuromodulation': self.use_neuromodulation,
            'attention_type': self.attention_type,
            'weight_decay': self.weight_decay,
            'use_batch_norm': self.use_batch_norm
        } 