"""
BioKANモデルの設定クラス
"""

from dataclasses import dataclass, field
from typing import List, Dict
from typing import Optional


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
    """BioKANモデルの設定を保持するデータクラス"""
    
    # モデル構造
    hidden_dim: int = 128
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # 学習設定
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-5
    
    # 神経伝達物質設定
    neurotransmitter: NeurotransmitterConfig = field(
        default_factory=lambda: NeurotransmitterConfig()
    )
    
    # アテンション設定
    attention_type: str = "biological"
    attention_dropout: float = 0.1
    
    # 転移学習設定
    transfer_learning: bool = False
    source_model_path: Optional[str] = None
    freeze_layers: int = 0
    
    def __post_init__(self):
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
        
        if not 0 <= self.weight_decay:
            raise ValueError("weight_decayは0以上である必要があります")
        
        valid_attention_types = ["biological", "cortical", "hierarchical"]
        if self.attention_type not in valid_attention_types:
            raise ValueError(
                f"attention_typeは{valid_attention_types}のいずれかである必要があります"
            )
        
        if not 0 <= self.attention_dropout <= 1:
            raise ValueError("attention_dropoutは0から1の間である必要があります")
        
        if self.freeze_layers < 0:
            raise ValueError("freeze_layersは0以上である必要があります")

    def to_dict(self) -> Dict:
        """設定をディクショナリ形式で返す"""
        return {
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'neurotransmitter': self.neurotransmitter.__dict__,
            'attention_type': self.attention_type,
            'attention_dropout': self.attention_dropout,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'transfer_learning': self.transfer_learning,
            'source_model_path': self.source_model_path,
            'freeze_layers': self.freeze_layers
        } 