"""
神経膠細胞（グリア細胞）の実装
"""

import numpy as np
from typing import Dict, Tuple, Optional

class Astrocyte:
    def __init__(self, region_shape: Tuple[int, ...], activation_threshold: float = 0.7):
        self.region_shape = region_shape
        self.activation_threshold = activation_threshold
        self.activation = np.zeros(region_shape)
        self.calcium_level = np.zeros(region_shape)
        
    def update(self, neural_activity: np.ndarray) -> None:
        self.activation = np.where(neural_activity > self.activation_threshold,
                                 neural_activity,
                                 self.activation * 0.95)
        
    def get_modulatory_effect(self) -> Dict[str, np.ndarray]:
        return {
            "excitation": self.activation * 1.2,
            "inhibition": 1.0 - self.activation * 0.8
        }

class Microglia:
    def __init__(self, region_shape: Tuple[int, ...]):
        self.region_shape = region_shape
        self.activation = np.zeros(region_shape)
        self.inflammatory_state = np.zeros(region_shape)
        
    def update(self, damage_signals: np.ndarray) -> None:
        self.activation = np.maximum(damage_signals, self.activation * 0.9)
        self.inflammatory_state = np.where(self.activation > 0.8,
                                         self.inflammatory_state + 0.1,
                                         self.inflammatory_state * 0.95)
        
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "activation": self.activation,
            "inflammation": self.inflammatory_state
        } 