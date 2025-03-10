"""
BioKANモデルの神経活動と説明可能性のための可視化ツール
神経伝達物質の動態、活性化パターン、アテンション分布などの可視化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import io
from PIL import Image


class NeuralActivityVisualizer:
    """
    BioKANモデルの神経活動を可視化するクラス
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid',
                color_palette: str = 'viridis'):
        """
        初期化
        
        Args:
            figsize: 図のサイズ
            style: seabornスタイル
            color_palette: 色パレット
        """
        self.figsize = figsize
        self.style = style
        self.color_palette = color_palette
        
        # スタイル設定
        sns.set_style(style)
        sns.set_palette(color_palette)
        
        # 生物学的な色マップの定義
        # - 興奮性活動: 赤→オレンジ
        # - 抑制性活動: 青→紫
        self.excitatory_cmap = LinearSegmentedColormap.from_list(
            'excitatory', ['#FFEBEE', '#FFCCBC', '#FFAB91', '#FF8A65', '#FF7043', '#FF5722', '#F4511E', '#E64A19', '#D84315', '#BF360C']
        )
        self.inhibitory_cmap = LinearSegmentedColormap.from_list(
            'inhibitory', ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1']
        )
        
        # 神経伝達物質の色マッピング
        self.neuromodulator_colors = {
            'dopamine': '#FF5722',     # オレンジ
            'serotonin': '#9C27B0',    # 紫
            'noradrenaline': '#2196F3', # 青
            'acetylcholine': '#4CAF50', # 緑
            'glutamate': '#F44336',    # 赤
            'gaba': '#3F51B5'          # 藍色
        }
    
    def plot_neuromodulator_dynamics(self, history: Dict[str, List[float]], 
                                   title: str = "神経伝達物質の動態") -> Figure:
        """
        神経伝達物質の時間的動態をプロット
        
        Args:
            history: 神経伝達物質の履歴データ
            title: グラフのタイトル
            
        Returns:
            生成された図
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(next(iter(history.values()))))
        
        for nt, values in history.items():
            color = self.neuromodulator_colors.get(nt, None)
            ax.plot(x, values, label=nt, linewidth=2, color=color)
        
        ax.set_xlabel('時間ステップ', fontsize=12)
        ax.set_ylabel('活性レベル', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # y軸の範囲を統一（-1.0から1.0）
        ax.set_ylim(-1.05, 1.05)
        
        # 0ラインの強調
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_activation_distribution(self, activations: torch.Tensor, 
                                   layer_name: str = "", 
                                   bins: int = 50) -> Figure:
        """
        活性化分布のヒストグラムをプロット
        
        Args:
            activations: 活性化テンソル
            layer_name: レイヤー名
            bins: ビン数
            
        Returns:
            生成された図
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy().flatten()
        else:
            activations = np.array(activations).flatten()
        
        # 3つの領域のデータを分離（負、ゼロ、正）
        neg_values = activations[activations < 0]
        zero_values = activations[activations == 0]
        pos_values = activations[activations > 0]
        
        # 全体のヒストグラム
        sns.histplot(activations, bins=bins, kde=True, ax=ax, alpha=0.3, color='gray')
        
        # 領域ごとのヒストグラム
        if len(neg_values) > 0:
            sns.histplot(neg_values, bins=bins, kde=True, ax=ax, alpha=0.6, color=self.neuromodulator_colors['gaba'], label='抑制性 (-)')
        
        if len(zero_values) > 0:
            sns.histplot(zero_values, bins=bins, kde=False, ax=ax, alpha=0.6, color='gray', label='不活性 (0)')
        
        if len(pos_values) > 0:
            sns.histplot(pos_values, bins=bins, kde=True, ax=ax, alpha=0.6, color=self.neuromodulator_colors['glutamate'], label='興奮性 (+)')
        
        # 統計情報
        mean_val = np.mean(activations)
        std_val = np.std(activations)
        median_val = np.median(activations)
        sparsity = np.mean(activations == 0)
        
        stats_text = f"平均: {mean_val:.4f}\n標準偏差: {std_val:.4f}\n中央値: {median_val:.4f}\nスパース度: {sparsity:.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('活性化値', fontsize=12)
        ax.set_ylabel('頻度', fontsize=12)
        ax.set_title(f"{layer_name}の活性化分布", fontsize=14)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_attention_heatmap(self, attention_weights: np.ndarray, 
                             token_labels: Optional[List[str]] = None,
                             head_idx: int = 0,
                             title: str = "アテンションヒートマップ") -> Figure:
        """
        アテンションウェイトをヒートマップとして可視化
        
        Args:
            attention_weights: アテンション重み行列 [head, seq_len, seq_len]
            token_labels: トークンのラベル（指定がなければインデックス）
            head_idx: 表示するアテンションヘッドのインデックス
            title: グラフのタイトル
            
        Returns:
            生成された図
        """
        # 指定されたヘッドのアテンション重みを取得
        if attention_weights.ndim > 2:
            attn = attention_weights[head_idx]
        else:
            attn = attention_weights
        
        seq_len = attn.shape[0]
        
        # ラベルが指定されていない場合はインデックスを使用
        if token_labels is None:
            token_labels = [f"Token {i}" for i in range(seq_len)]
        
        # トークン数がラベル数と一致しない場合は調整
        if len(token_labels) != seq_len:
            token_labels = token_labels[:seq_len] if len(token_labels) > seq_len else token_labels + [f"Token {i}" for i in range(len(token_labels), seq_len)]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # ヒートマップの描画
        sns.heatmap(attn, annot=False, cmap='viridis', ax=ax, 
                   xticklabels=token_labels, yticklabels=token_labels)
        
        # ラベルが長い場合は回転
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.set_xlabel('宛先トークン', fontsize=12)
        ax.set_ylabel('ソーストークン', fontsize=12)
        ax.set_title(f"{title} (ヘッド {head_idx})", fontsize=14)
        
        # エントロピーと集中度の計算
        entropy = -np.sum(attn * np.log(np.clip(attn, 1e-10, 1.0)), axis=1).mean()
        
        # 集中度（ジニ係数風）の計算
        sorted_attn = np.sort(attn, axis=1)
        cumsum = np.cumsum(sorted_attn, axis=1)
        indices = np.arange(1, seq_len + 1)
        ideal_cumsum = indices / seq_len
        concentration = np.abs(cumsum - ideal_cumsum.reshape(1, -1)).mean()
        
        stats_text = f"エントロピー: {entropy:.4f}\n集中度: {concentration:.4f}"
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_multi_head_attention(self, attention_weights: np.ndarray,
                                token_labels: Optional[List[str]] = None,
                                title: str = "マルチヘッドアテンション") -> Figure:
        """
        複数のアテンションヘッドを可視化
        
        Args:
            attention_weights: アテンション重み行列 [head, seq_len, seq_len]
            token_labels: トークンのラベル
            title: グラフのタイトル
            
        Returns:
            生成された図
        """
        num_heads = attention_weights.shape[0]
        
        # グリッドサイズの計算（近似的な正方形）
        grid_size = int(np.ceil(np.sqrt(num_heads)))
        
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(self.figsize[0]*1.5, self.figsize[1]*1.5))
        axs = axs.flatten()
        
        for h in range(num_heads):
            if h < len(axs):
                ax = axs[h]
                
                # 各ヘッドのヒートマップ
                sns.heatmap(attention_weights[h], cmap='viridis', 
                           annot=False, cbar=False, ax=ax)
                
                ax.set_title(f"ヘッド {h}")
                
                # ラベルと目盛りは主要なヘッドのみに表示（スペース節約）
                if h % grid_size == 0:  # 左端のプロット
                    if token_labels is not None:
                        ax.set_yticklabels(token_labels, fontsize=8)
                    ax.set_ylabel('ソース')
                else:
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                
                if h >= (grid_size * (grid_size - 1)):  # 下端のプロット
                    if token_labels is not None:
                        ax.set_xticklabels(token_labels, fontsize=8, rotation=45, ha='right')
                    ax.set_xlabel('宛先')
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
        
        # 使用していないサブプロットを非表示
        for h in range(num_heads, len(axs)):
            axs[h].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # タイトル用にスペースを確保
        
        return fig
    
    def create_activation_animation(self, activation_sequence: List[np.ndarray],
                                  layer_name: str = "",
                                  interval: int = 200,
                                  colormap: str = 'viridis') -> animation.FuncAnimation:
        """
        活性化の時間変化をアニメーションとして作成
        
        Args:
            activation_sequence: 時系列の活性化データ
            layer_name: レイヤー名
            interval: フレーム間隔（ミリ秒）
            colormap: カラーマップ
            
        Returns:
            アニメーションオブジェクト
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 活性化値の最大値と最小値を取得（スケーリング用）
        all_data = np.concatenate([act.flatten() for act in activation_sequence])
        vmin, vmax = np.min(all_data), np.max(all_data)
        
        # 最初のフレームのヒートマップを表示
        im = ax.imshow(activation_sequence[0], cmap=colormap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        
        title = ax.set_title(f"{layer_name} - フレーム 0")
        
        def update(frame):
            """アニメーションの各フレームを更新"""
            im.set_array(activation_sequence[frame])
            title.set_text(f"{layer_name} - フレーム {frame}")
            return im, title
        
        ani = animation.FuncAnimation(fig, update, frames=len(activation_sequence),
                                    interval=interval, blit=True)
        
        plt.tight_layout()
        return ani
    
    def plot_neuromodulator_effects(self, effects: Dict[str, float], 
                                 title: str = "神経伝達物質効果") -> Figure:
        """
        神経伝達物質の効果を可視化
        
        Args:
            effects: 神経伝達物質とその効果値
            title: グラフのタイトル
            
        Returns:
            生成された図
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 効果を昇順でソート
        sorted_effects = sorted(effects.items(), key=lambda x: x[1])
        names, values = zip(*sorted_effects)
        
        # 正と負の効果で色分け
        colors = [self.neuromodulator_colors.get(n, '#999999') for n in names]
        
        bars = ax.barh(names, values, color=colors)
        
        # ゼロラインを追加
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # 値を棒グラフの端に表示
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 if width > 0 else width - 0.01
            label_ha = 'left' if width > 0 else 'right'
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                   va='center', ha=label_ha, fontsize=10)
        
        ax.set_xlabel('効果の強さ', fontsize=12)
        ax.set_ylabel('神経伝達物質', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_layer_comparison(self, layer_stats: Dict[str, Dict[str, float]], 
                           metric: str = 'mean',
                           title: str = "レイヤー比較") -> Figure:
        """
        複数レイヤーの特定の指標を比較
        
        Args:
            layer_stats: レイヤーごとの統計情報
            metric: 比較する指標 ('mean', 'std', 'sparsity', 'max', 'min' など)
            title: グラフのタイトル
            
        Returns:
            生成された図
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        layers = []
        values = []
        
        for layer_name, stats in layer_stats.items():
            if metric in stats:
                layers.append(layer_name)
                values.append(stats[metric])
        
        # レイヤー名が長いとき、表示を調整
        if max([len(l) for l in layers]) > 15:
            plt.figure(figsize=(self.figsize[0], self.figsize[1] * 1.5))
            ax.barh(layers, values)
            ax.set_xlabel(f'{metric}の値', fontsize=12)
            ax.set_ylabel('レイヤー', fontsize=12)
        else:
            ax.bar(layers, values)
            ax.set_xlabel('レイヤー', fontsize=12)
            ax.set_ylabel(f'{metric}の値', fontsize=12)
            plt.xticks(rotation=45, ha='right')
        
        ax.set_title(f"{title} - {metric}", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def visualize_explanation(explanation_result: Dict[str, Any], 
                         figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
    """
    BioKANExplainerの結果を可視化する関数
    
    Args:
        explanation_result: 説明結果
        figsize: 図のサイズ
        
    Returns:
        図のディクショナリ
    """
    visualizer = NeuralActivityVisualizer(figsize=figsize)
    figures = {}
    
    # 予測結果の可視化
    if 'predictions' in explanation_result:
        fig, ax = plt.subplots(figsize=figsize)
        preds = explanation_result['predictions']
        probs = explanation_result.get('probabilities', None)
        
        if probs is not None and len(probs.shape) > 1 and probs.shape[1] > 1:
            # 多クラス分類の場合
            num_classes = probs.shape[1]
            bar_positions = np.arange(num_classes)
            
            # 最初のサンプルのみ表示
            sample_probs = probs[0]
            bars = ax.bar(bar_positions, sample_probs)
            
            # 予測クラスを強調
            pred_class = preds[0]
            bars[pred_class].set_color('red')
            
            ax.set_xticks(bar_positions)
            ax.set_xlabel('クラス', fontsize=12)
            ax.set_ylabel('確率', fontsize=12)
            ax.set_title('予測確率', fontsize=14)
            
            # 確率値を表示
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        else:
            # 二値分類または回帰の場合
            ax.plot(preds, marker='o', linestyle='-', markersize=8)
            
            if probs is not None:
                ax.plot(probs, marker='x', linestyle='--', label='確率/スコア')
                ax.legend()
            
            ax.set_xlabel('サンプル', fontsize=12)
            ax.set_ylabel('予測/確率', fontsize=12)
            ax.set_title('予測結果', fontsize=14)
        
        figures['predictions'] = fig
    
    # 神経伝達物質の分析
    if 'neuromodulator_analysis' in explanation_result:
        neuro_analysis = explanation_result['neuromodulator_analysis']
        
        # 各神経伝達物質の要約
        for nt, data in neuro_analysis.items():
            if 'values_by_layer' in data:
                # レイヤーごとの神経伝達物質値をプロット
                layer_names = [item['layer'] for item in data['values_by_layer']]
                values = [item['value'] for item in data['values_by_layer']]
                
                fig, ax = plt.subplots(figsize=figsize)
                
                # レイヤー名が長い場合は横向きのバープロット
                if max([len(l) for l in layer_names]) > 15:
                    bars = ax.barh(layer_names, values, color=visualizer.neuromodulator_colors.get(nt, '#999999'))
                    ax.set_xlabel(f'{nt}の値', fontsize=12)
                    ax.set_ylabel('レイヤー', fontsize=12)
                else:
                    bars = ax.bar(layer_names, values, color=visualizer.neuromodulator_colors.get(nt, '#999999'))
                    ax.set_xlabel('レイヤー', fontsize=12)
                    ax.set_ylabel(f'{nt}の値', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                
                ax.set_title(f'{nt}のレイヤー別分布', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                # ゼロラインを追加
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                figures[f'neuromodulator_{nt}'] = fig
    
    # レイヤーの活性化統計
    if 'layer_activations' in explanation_result:
        layer_activations = explanation_result['layer_activations']
        
        # 平均活性化のレイヤー比較
        fig = visualizer.plot_layer_comparison(layer_activations, metric='mean', 
                                             title='レイヤー別平均活性化')
        figures['layer_mean_activations'] = fig
        
        # スパース度のレイヤー比較
        fig = visualizer.plot_layer_comparison(layer_activations, metric='sparsity', 
                                             title='レイヤー別スパース度')
        figures['layer_sparsity'] = fig
    
    # アテンションパターンの可視化
    if 'attention_patterns' in explanation_result:
        attention_patterns = explanation_result['attention_patterns']
        
        for layer_name, attn_data in attention_patterns.items():
            if 'weights' in attn_data:
                weights = attn_data['weights']
                
                # 単一ヘッドの場合
                if len(weights.shape) == 2 or weights.shape[0] == 1:
                    if len(weights.shape) > 2:
                        weights = weights[0]  # 最初のヘッド
                        
                    fig = visualizer.plot_attention_heatmap(
                        weights, title=f'{layer_name}のアテンション')
                    figures[f'attention_{layer_name}_head0'] = fig
                
                # マルチヘッドの場合
                elif len(weights.shape) > 2 and weights.shape[0] > 1:
                    fig = visualizer.plot_multi_head_attention(
                        weights, title=f'{layer_name}のマルチヘッドアテンション')
                    figures[f'attention_{layer_name}_all_heads'] = fig
    
    return figures


def plot_statistical_comparison(features: np.ndarray, 
                             labels: np.ndarray, 
                             neural_states: Dict[str, np.ndarray],
                             feature_labels: Optional[List[str]] = None) -> Dict[str, Figure]:
    """
    特徴量と神経状態の統計的比較を可視化
    
    Args:
        features: 入力特徴量
        labels: ラベル
        neural_states: 神経状態（神経伝達物質など）
        feature_labels: 特徴量のラベル
        
    Returns:
        図のディクショナリ
    """
    figures = {}
    
    # クラス別の特徴量分布
    fig, ax = plt.subplots(figsize=(12, 8))
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        class_features = features[labels == label]
        mean_features = np.mean(class_features, axis=0)
        std_features = np.std(class_features, axis=0)
        
        x = np.arange(len(mean_features))
        ax.errorbar(x, mean_features, yerr=std_features, fmt='o-', label=f'クラス {label}')
    
    if feature_labels:
        ax.set_xticks(np.arange(len(feature_labels)))
        ax.set_xticklabels(feature_labels, rotation=45, ha='right')
    
    ax.set_xlabel('特徴量', fontsize=12)
    ax.set_ylabel('平均値', fontsize=12)
    ax.set_title('クラス別特徴量分布', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figures['feature_distribution'] = fig
    
    # 神経状態と予測の相関
    for nt_name, nt_values in neural_states.items():
        if nt_values.ndim == 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for label in unique_labels:
                nt_by_class = nt_values[labels == label]
                ax.hist(nt_by_class, bins=20, alpha=0.5, label=f'クラス {label}')
            
            ax.set_xlabel(f'{nt_name}の値', fontsize=12)
            ax.set_ylabel('頻度', fontsize=12)
            ax.set_title(f'{nt_name}のクラス別分布', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            figures[f'neuro_distribution_{nt_name}'] = fig
    
    # 主成分分析を使用して特徴量と神経状態の関係を可視化
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # 特徴量の標準化とPCA
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(features)
        
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(standardized_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # クラスごとの散布図
        for label in unique_labels:
            mask = labels == label
            ax.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                     label=f'クラス {label}', alpha=0.7)
        
        # 神経伝達物質値でポイントの大きさを変える
        for nt_name, nt_values in neural_states.items():
            if nt_values.ndim == 1:
                # 値を0-1にスケーリング
                scaled_values = (nt_values - np.min(nt_values)) / (np.max(nt_values) - np.min(nt_values) + 1e-10)
                sizes = 20 + scaled_values * 100
                
                fig_nt, ax_nt = plt.subplots(figsize=(12, 8))
                
                scatter = ax_nt.scatter(pca_features[:, 0], pca_features[:, 1], 
                                      c=nt_values, s=sizes, cmap='viridis', alpha=0.7)
                
                plt.colorbar(scatter, ax=ax_nt, label=f'{nt_name}の値')
                
                ax_nt.set_xlabel('主成分1', fontsize=12)
                ax_nt.set_ylabel('主成分2', fontsize=12)
                ax_nt.set_title(f'PCA特徴量空間での{nt_name}分布', fontsize=14)
                
                plt.tight_layout()
                figures[f'pca_{nt_name}'] = fig_nt
        
        ax.set_xlabel('主成分1', fontsize=12)
        ax.set_ylabel('主成分2', fontsize=12)
        ax.set_title('クラス別PCA特徴量空間', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['pca_features'] = fig
        
    except ImportError:
        print("scikit-learnがインストールされていないため、PCA可視化をスキップします")
    
    return figures 