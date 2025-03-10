"""
実験結果を分析するユーティリティクラス
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path

from biokan.utils.visualization_utils import setup_japanese_fonts, set_plot_style


class ExperimentAnalyzer:
    """実験結果を分析するクラス"""
    
    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: 実験結果が保存されているディレクトリ
        """
        self.results_dir = Path(results_dir)
        setup_japanese_fonts()
        set_plot_style()
    
    def load_experiment_data(self, experiment_name: str) -> Dict:
        """実験データを読み込む

        Args:
            experiment_name: 実験名

        Returns:
            実験データを含む辞書
        """
        data_path = self.results_dir / experiment_name / "results.json"
        if not data_path.exists():
            raise FileNotFoundError(f"実験データが見つかりません: {data_path}")
        
        with open(data_path) as f:
            data = json.load(f)
        return data
    
    def compare_learning_curves(
        self,
        experiment_names: List[str],
        metric: str = "accuracy",
        title: Optional[str] = None
    ) -> None:
        """学習曲線を比較する

        Args:
            experiment_names: 比較する実験名のリスト
            metric: 比較する指標（"accuracy"または"loss"）
            title: グラフのタイトル（オプション）
        """
        plt.figure(figsize=(10, 6))
        
        for exp_name in experiment_names:
            data = self.load_experiment_data(exp_name)
            if metric == "accuracy":
                train_metric = data["train_accuracies"]
                val_metric = data["val_accuracies"]
                ylabel = "精度 (%)"
            else:
                train_metric = data["train_losses"]
                val_metric = data["val_losses"]
                ylabel = "損失"
            
            epochs = range(1, len(train_metric) + 1)
            plt.plot(epochs, train_metric, '-', label=f"{exp_name} (訓練)")
            plt.plot(epochs, val_metric, '--', label=f"{exp_name} (検証)")
        
        plt.xlabel("エポック")
        plt.ylabel(ylabel)
        plt.title(title or f"{metric}の比較")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def generate_comprehensive_report(
        self,
        experiment_names: List[str],
        output_dir: str
    ) -> None:
        """包括的な分析レポートを生成する

        Args:
            experiment_names: 分析する実験名のリスト
            output_dir: レポートの出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 結果の要約を作成
        summary = []
        for exp_name in experiment_names:
            data = self.load_experiment_data(exp_name)
            summary.append({
                "実験名": exp_name,
                "最良の検証精度": f"{data['best_val_accuracy']:.2f}%",
                "最終訓練損失": f"{data['train_losses'][-1]:.4f}",
                "最終検証損失": f"{data['val_losses'][-1]:.4f}",
                "エポック数": len(data["train_losses"]),
                "学習率": data.get("learning_rate", "N/A"),
                "バッチサイズ": data.get("batch_size", "N/A")
            })
        
        # 要約をCSVとして保存
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_path / "summary.csv", index=False)
        
        # 学習曲線のプロット
        self.compare_learning_curves(
            experiment_names,
            metric="accuracy",
            title="精度の比較"
        )
        plt.savefig(output_path / "accuracy_comparison.png")
        plt.close()
        
        self.compare_learning_curves(
            experiment_names,
            metric="loss",
            title="損失の比較"
        )
        plt.savefig(output_path / "loss_comparison.png")
        plt.close()
        
        # 詳細な分析レポートを作成
        report_path = output_path / "report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== 実験分析レポート ===\n\n")
            
            for exp_name in experiment_names:
                data = self.load_experiment_data(exp_name)
                f.write(f"\n== {exp_name} ==\n")
                f.write(f"最良の検証精度: {data['best_val_accuracy']:.2f}%\n")
                f.write(f"訓練エポック数: {len(data['train_losses'])}\n")
                f.write(f"最終訓練損失: {data['train_losses'][-1]:.4f}\n")
                f.write(f"最終検証損失: {data['val_losses'][-1]:.4f}\n")
                
                if "hyperparameters" in data:
                    f.write("\nハイパーパラメータ:\n")
                    for key, value in data["hyperparameters"].items():
                        f.write(f"  {key}: {value}\n")
                
                f.write("\n")
        
        print(f"分析レポートが {output_path} に生成されました。")
    
    def analyze_hyperparameter_sensitivity(
        self,
        experiment_name: str,
        param_name: str
    ) -> None:
        """ハイパーパラメータの感度分析を行う

        Args:
            experiment_name: 実験名
            param_name: 分析するハイパーパラメータ名
        """
        data = self.load_experiment_data(experiment_name)
        if "hyperparameter_search" not in data:
            raise ValueError("ハイパーパラメータ探索の結果が見つかりません")
        
        search_results = data["hyperparameter_search"]
        param_values = []
        accuracies = []
        
        for trial in search_results:
            if param_name in trial["params"]:
                param_values.append(trial["params"][param_name])
                accuracies.append(trial["accuracy"])
        
        if not param_values:
            raise ValueError(f"パラメータ {param_name} のデータが見つかりません")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(param_values, accuracies, alpha=0.6)
        plt.xlabel(param_name)
        plt.ylabel("検証精度 (%)")
        plt.title(f"{param_name}の感度分析")
        
        # 傾向線の追加
        z = np.polyfit(param_values, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(param_values, p(param_values), "r--", alpha=0.8)
        
        plt.grid(True)
        plt.show() 