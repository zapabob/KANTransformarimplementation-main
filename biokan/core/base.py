"""
BioKANの基本クラスと抽象クラスを定義
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

class BioKANBase(nn.Module, ABC):
    """
    BioKANモデルの基本クラス
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
        
    @abstractmethod
    def update_neuromodulation(self, activity: torch.Tensor) -> None:
        pass
        
class BiologicalLayerBase(nn.Module, ABC):
    """
    生物学的レイヤーの基本クラス
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    @abstractmethod
    def compute_activation(self, x: torch.Tensor) -> torch.Tensor:
        pass
        
    @abstractmethod
    def update_state(self, activity: torch.Tensor) -> None:
        pass 