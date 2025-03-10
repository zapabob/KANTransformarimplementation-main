"""
MNIST実験の実行スクリプト
"""

import torch
import argparse
from pathlib import Path

from biokan.core.config import BioKANConfig
from biokan.experiments.mnist.mnist_experiment import MNISTExperiment
from biokan.experiments.fashion_mnist.fashion_mnist_experiment import FashionMNISTExperiment
from biokan.experiments.analyzer import ExperimentAnalyzer
from biokan.experiments.hyperparameter_optimizer import HyperparameterOptimizer

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='MNIST実験の実行')
    parser.add_argument('--epochs', type=int, default=100,
                      help='訓練エポック数')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='学習率')
    parser.add_argument('--hidden-dim', type=int, default=512,
                      help='隠れ層の次元')
    parser.add_argument('--num-layers', type=int, default=6,
                      help='レイヤー数')
    args = parser.parse_args()
    
    # 設定の作成
    config = BioKANConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    # 実験の実行
    experiment = MNISTExperiment(config)
    experiment.run(args.epochs)
    
    # 実験結果の分析
    analyzer = ExperimentAnalyzer("results")
    analyzer.compare_learning_curves(
        ["mnist_experiment", "fashion_mnist_experiment"],
        metric="accuracy"
    )
    
    analyzer.generate_comprehensive_report(
        ["mnist_experiment", "fashion_mnist_experiment"],
        "analysis_results"
    )
    
    optimizer = HyperparameterOptimizer(
        MNISTExperiment,
        n_trials=100,
        study_name="mnist_optimization"
    )
    best_params = optimizer.optimize()
    
if __name__ == '__main__':
    main() 