"""
実験管理の基本クラス
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from pathlib import Path
import json
import logging
from datetime import datetime

from ..core.config import BioKANConfig

class BaseExperiment(ABC):
    """実験の基本クラス"""
    
    def __init__(self, config: BioKANConfig, experiment_name: str):
        self.config = config
        self.experiment_name = experiment_name
        self.results_dir = Path("results") / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # ロギングの設定
        self.setup_logging()
        
        # 実験の状態
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
        
    def setup_logging(self):
        """ロギングの設定"""
        log_file = self.results_dir / f"{self.experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)
        
    @abstractmethod
    def prepare_data(self):
        """データの準備"""
        pass
        
    @abstractmethod
    def create_model(self):
        """モデルの作成"""
        pass
        
    @abstractmethod
    def train_epoch(self):
        """1エポックの訓練"""
        pass
        
    @abstractmethod
    def validate(self):
        """検証"""
        pass
        
    def save_checkpoint(self, is_best: bool = False):
        """チェックポイントの保存"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        # 通常のチェックポイント
        checkpoint_path = self.results_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # ベストモデル
        if is_best:
            best_path = self.results_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイントの読み込み"""
        checkpoint = torch.load(checkpoint_path)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint['history']
        
    def save_results(self):
        """実験結果の保存"""
        results = {
            'config': self.config.to_dict(),
            'history': self.history,
            'best_metric': self.best_metric,
            'final_epoch': self.current_epoch
        }
        
        results_file = self.results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
    def log_metrics(self, metrics: Dict[str, float], phase: str):
        """メトリクスのロギング"""
        message = f"Epoch {self.current_epoch} - {phase}: "
        message += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(message)
        
        # 履歴の更新
        for k, v in metrics.items():
            if k not in self.history['metrics']:
                self.history['metrics'][k] = []
            self.history['metrics'][k].append(v) 