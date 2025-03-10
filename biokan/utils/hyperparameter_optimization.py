"""
ハイパーパラメータ最適化ツール
"""

import optuna
from optuna.trial import Trial
import torch
import json
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import logging

from ..core.config import BioKANConfig
from ..experiments.base_experiment import BaseExperiment

class HyperparameterOptimizer:
    """ハイパーパラメータ最適化クラス"""
    
    def __init__(self, 
                 experiment_class: type,
                 n_trials: int = 100,
                 study_name: str = "biokan_optimization",
                 storage: Optional[str] = None):
        self.experiment_class = experiment_class
        self.n_trials = n_trials
        self.study_name = study_name
        
        # Optunaの設定
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True
        )
        
        # ロギングの設定
        self.setup_logging()
        
    def setup_logging(self):
        """ロギングの設定"""
        log_file = Path("results") / "optimization" / f"{self.study_name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.study_name)
        
    def define_search_space(self, trial: Trial) -> Dict[str, Any]:
        """探索空間の定義"""
        config = {
            'hidden_dim': trial.suggest_int('hidden_dim', 64, 512),
            'num_layers': trial.suggest_int('num_layers', 2, 8),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            
            # 神経伝達物質の設定
            'neurotransmitter': {
                'baseline_levels': [
                    trial.suggest_float(f'nt_baseline_{i}', 0.5, 2.0)
                    for i in range(6)
                ],
                'release_rates': [
                    trial.suggest_float(f'nt_release_{i}', 0.05, 0.3)
                    for i in range(6)
                ],
                'reuptake_rates': [
                    trial.suggest_float(f'nt_reuptake_{i}', 0.1, 0.4)
                    for i in range(6)
                ]
            },
            
            # アテンションの設定
            'attention': {
                'num_heads': trial.suggest_int('attention_heads', 4, 16),
                'head_dim': trial.suggest_int('head_dim', 32, 128),
                'dropout': trial.suggest_float('attention_dropout', 0.1, 0.3)
            },
            
            # 学習設定
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'num_epochs': trial.suggest_int('num_epochs', 50, 200)
        }
        
        return config
        
    def objective(self, trial: Trial) -> float:
        """最適化の目的関数"""
        # 設定の生成
        config = BioKANConfig(**self.define_search_space(trial))
        
        try:
            # 実験の実行
            experiment = self.experiment_class(config)
            experiment.run(config.num_epochs)
            
            # 結果の取得
            best_metric = experiment.best_metric
            
            # 結果のロギング
            self.logger.info(f"Trial {trial.number}: best_metric = {best_metric}")
            
            return best_metric
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()
            
    def optimize(self) -> Dict:
        """最適化の実行"""
        self.logger.info(f"Starting optimization with {self.n_trials} trials")
        
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=None
        )
        
        # 最適なパラメータの取得
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        self.logger.info(f"Optimization finished")
        self.logger.info(f"Best value: {best_value}")
        self.logger.info(f"Best parameters: {best_params}")
        
        # 結果の保存
        results = {
            'best_value': best_value,
            'best_params': best_params,
            'study_name': self.study_name
        }
        
        results_file = Path("results") / "optimization" / f"{self.study_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        return results
        
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """最適化の履歴をプロット"""
        fig = optuna.visualization.plot_optimization_history(self.study)
        if save_path:
            fig.write_image(save_path)
            
    def plot_parameter_importances(self, save_path: Optional[str] = None):
        """パラメータの重要度をプロット"""
        fig = optuna.visualization.plot_param_importances(self.study)
        if save_path:
            fig.write_image(save_path)
            
    def plot_parallel_coordinate(self, save_path: Optional[str] = None):
        """パラレル座標プロットの生成"""
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        if save_path:
            fig.write_image(save_path) 