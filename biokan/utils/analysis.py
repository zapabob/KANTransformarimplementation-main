"""
実験結果の比較分析ツール
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json
from scipy import stats

class ExperimentAnalyzer:
    """実験結果の分析クラス"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
    def load_multiple_results(self, experiment_names: List[str]) -> Dict[str, Dict]:
        """複数の実験結果を読み込む"""
        results = {}
        for name in experiment_names:
            results_file = self.results_dir / name / "results.json"
            with open(results_file, 'r') as f:
                results[name] = json.load(f)
        return results
        
    def compare_learning_curves(self, experiment_names: List[str],
                              metric: str = 'accuracy',
                              save_path: Optional[str] = None):
        """学習曲線の比較"""
        results = self.load_multiple_results(experiment_names)
        
        plt.figure(figsize=(12, 6))
        for name, result in results.items():
            if metric in result['history']['metrics']:
                values = result['history']['metrics'][metric]
                epochs = range(1, len(values) + 1)
                plt.plot(epochs, values, label=name)
                
        plt.title(f'Learning Curves Comparison - {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def statistical_analysis(self, experiment_names: List[str],
                           metric: str = 'accuracy') -> pd.DataFrame:
        """統計分析"""
        results = self.load_multiple_results(experiment_names)
        stats_data = []
        
        for name, result in results.items():
            if metric in result['history']['metrics']:
                values = result['history']['metrics'][metric]
                stats_dict = {
                    'experiment': name,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1],
                    'best': max(values)
                }
                stats_data.append(stats_dict)
                
        return pd.DataFrame(stats_data)
        
    def parameter_correlation_analysis(self, experiment_results: List[Dict],
                                    target_metric: str = 'best_metric',
                                    save_path: Optional[str] = None):
        """パラメータと性能の相関分析"""
        # データの準備
        data = []
        for result in experiment_results:
            params = result['config']
            metric_value = result[target_metric]
            
            flat_params = {}
            for k, v in params.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat_params[f"{k}_{sub_k}"] = sub_v
                else:
                    flat_params[k] = v
                    
            flat_params[target_metric] = metric_value
            data.append(flat_params)
            
        df = pd.DataFrame(data)
        
        # 相関分析
        correlations = df.corr()[target_metric].sort_values(ascending=False)
        
        # 可視化
        plt.figure(figsize=(12, 6))
        correlations[1:11].plot(kind='bar')
        plt.title('Parameter Correlation Analysis')
        plt.xlabel('Parameter')
        plt.ylabel(f'Correlation with {target_metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
        return correlations
        
    def performance_distribution(self, experiment_names: List[str],
                               metric: str = 'accuracy',
                               save_path: Optional[str] = None):
        """性能分布の分析"""
        results = self.load_multiple_results(experiment_names)
        
        plt.figure(figsize=(12, 6))
        data = []
        labels = []
        
        for name, result in results.items():
            if metric in result['history']['metrics']:
                values = result['history']['metrics'][metric]
                data.append(values)
                labels.append(name)
                
        plt.boxplot(data, labels=labels)
        plt.title(f'Performance Distribution - {metric}')
        plt.ylabel(metric)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def convergence_analysis(self, experiment_names: List[str],
                           metric: str = 'loss',
                           window_size: int = 5,
                           save_path: Optional[str] = None):
        """収束性の分析"""
        results = self.load_multiple_results(experiment_names)
        
        plt.figure(figsize=(12, 6))
        for name, result in results.items():
            if metric in result['history']['metrics']:
                values = result['history']['metrics'][metric]
                
                # 移動平均の計算
                smoothed = pd.Series(values).rolling(window=window_size).mean()
                epochs = range(1, len(values) + 1)
                
                plt.plot(epochs, smoothed, label=f'{name} (MA{window_size})')
                
        plt.title(f'Convergence Analysis - {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(f'Smoothed {metric}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def generate_comprehensive_report(self, experiment_names: List[str],
                                   output_dir: str):
        """総合的な分析レポートの生成"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 学習曲線の比較
        self.compare_learning_curves(
            experiment_names,
            save_path=output_path / "learning_curves.png"
        )
        
        # 2. 統計分析
        stats_df = self.statistical_analysis(experiment_names)
        stats_df.to_csv(output_path / "statistical_analysis.csv")
        
        # 3. 性能分布の分析
        self.performance_distribution(
            experiment_names,
            save_path=output_path / "performance_distribution.png"
        )
        
        # 4. 収束性の分析
        self.convergence_analysis(
            experiment_names,
            save_path=output_path / "convergence_analysis.png"
        )
        
        # 5. レポートの生成
        report = {
            'experiments': experiment_names,
            'statistics': stats_df.to_dict(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path / "analysis_report.json", 'w') as f:
            json.dump(report, f, indent=4) 