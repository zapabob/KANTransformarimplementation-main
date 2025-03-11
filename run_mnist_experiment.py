"""
MNIST実験の実行スクリプト
"""

import torch
import argparse
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import numpy as np

from biokan.core.config import BioKANConfig
from biokan.experiments.mnist.mnist_experiment import MNISTExperiment
# from biokan.experiments.analyzer import ExperimentAnalyzer  # 一時的に無効化
from biokan.experiments.analysis.model_analysis import BioKANAnalyzer

class EarlyStopping:
    """早期停止の実装"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        
    def __call__(self, val_loss, model, save_path='checkpoint.pt'):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, save_path)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'早期停止カウンター: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, save_path):
        if self.verbose:
            print(f'検証損失が減少しました ({self.val_loss_min:.6f} --> {val_loss:.6f}). モデルを保存中...')
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='MNIST実験の実行')
    parser.add_argument('--epochs', type=int, default=50,
                      help='訓練エポック数')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=2e-4,
                      help='初期学習率')
    parser.add_argument('--hidden-dim', type=int, default=512,
                      help='隠れ層の次元')
    parser.add_argument('--num-layers', type=int, default=6,
                      help='レイヤー数')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='ドロップアウト率')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                      help='重み減衰（L2正則化）')
    parser.add_argument('--patience', type=int, default=5,
                      help='早期停止の待機エポック数')
    parser.add_argument('--use-wandb', action='store_true',
                      help='Weights & Biasesを使用する')
    parser.add_argument('--analyze', action='store_true',
                      help='モデルの詳細な分析を実行する')
    parser.add_argument('--use-batch-norm', action='store_true', default=True,
                      help='バッチ正規化を使用する')
    parser.add_argument('--scheduler-type', type=str, default='cosine',
                      choices=['cosine', 'step', 'none'],
                      help='学習率スケジューラの種類')
    args = parser.parse_args()
    
    # GPU利用可能性の確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用デバイス: {device}')
    
    if args.use_wandb:
        wandb.init(
            project="biokan-mnist",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "use_batch_norm": args.use_batch_norm,
                "scheduler_type": args.scheduler_type
            }
        )
    
    # 設定の作成
    config = BioKANConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        use_neuromodulation=True,
        use_batch_norm=args.use_batch_norm,
        device=device
    )
    
    # 実験の実行
    experiment = MNISTExperiment(config)
    analyzer = BioKANAnalyzer(experiment.model, device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # 学習率スケジューラの設定
    if args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            experiment.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    elif args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            experiment.optimizer,
            step_size=10,
            gamma=0.5
        )
    else:
        scheduler = None
    
    print('実験を開始します...')
    
    # 訓練ループ
    for epoch in range(args.epochs):
        # 訓練
        train_metrics = experiment.train_epoch()
        analyzer.history['train_loss'].append(train_metrics['loss'])
        analyzer.history['train_acc'].append(train_metrics['accuracy'])
        
        # 検証
        val_metrics = experiment.validate()
        analyzer.history['val_loss'].append(val_metrics['loss'])
        analyzer.history['val_acc'].append(val_metrics['accuracy'])
        
        # 学習率の更新
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'現在の学習率: {current_lr:.2e}')
        
        # 早期停止の判定
        early_stopping(val_metrics['loss'], experiment.model)
        if early_stopping.early_stop:
            print('早期停止が発動しました。訓練を終了します。')
            break
        
        # 神経調節の状態を記録
        if experiment.model.blocks[0].use_neuromodulation:
            states = [block.neuromodulation.get_state() for block in experiment.model.blocks]
            analyzer.history['neuromodulation_states'].append(states)
        
        # 進捗の表示
        print(f'エポック {epoch+1}/{args.epochs}:')
        print(f'  訓練損失: {train_metrics["loss"]:.4f}, 訓練精度: {train_metrics["accuracy"]:.2f}%')
        print(f'  検証損失: {val_metrics["loss"]:.4f}, 検証精度: {val_metrics["accuracy"]:.2f}%')
    
    print('実験が完了しました。分析を開始します...')
    
    # 最良のモデルを読み込み
    experiment.model.load_state_dict(torch.load('checkpoint.pt'))
    
    # 学習曲線の可視化
    analyzer.plot_learning_curves(save_path='learning_curves.png')
    
    if args.analyze:
        # クラスごとの性能分析
        mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        class_accuracy = analyzer.analyze_class_performance(
            experiment.test_loader,
            class_names=mnist_classes
        )
        print("\nクラスごとの精度:")
        for cls, acc in enumerate(class_accuracy):
            print(f"数字 {cls}: {acc:.2f}%")
        
        # 神経調節の効果分析
        if experiment.model.blocks[0].use_neuromodulation:
            neuromod_states = analyzer.analyze_neuromodulation(experiment.test_loader)
            
            # 神経調節の状態をファイルに保存
            import json
            with open('neuromodulation_analysis.json', 'w') as f:
                json.dump(
                    {f'block_{i}': state for i, state in enumerate(neuromod_states)},
                    f,
                    indent=2
                )
        
        # アテンションパターンの分析
        analyzer.analyze_attention_patterns(experiment.test_loader)
        
        # 特徴空間の可視化
        analyzer.visualize_feature_space(experiment.test_loader)
        
        # 転移学習の準備例（CIFAR-10用）
        print('転移学習の準備...')
        analyzer.prepare_transfer_learning(new_num_classes=10)
    
    if args.use_wandb:
        wandb.finish()
    
if __name__ == '__main__':
    main() 