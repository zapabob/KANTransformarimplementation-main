"""
高度な可視化機能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import networkx as nx

class AdvancedVisualizer:
    """高度な可視化クラス"""
    
    def __init__(self):
        self.neurotransmitter_colors = {
            'glutamate': '#FF6B6B',
            'gaba': '#4ECDC4',
            'dopamine': '#45B7D1',
            'serotonin': '#96CEB4',
            'noradrenaline': '#FFEEAD',
            'acetylcholine': '#D4A5A5'
        }
        
    def plot_neurotransmitter_dynamics(self, 
                                     nt_levels: Dict[str, List[float]],
                                     time_points: List[float],
                                     save_path: Optional[str] = None):
        """
        神経伝達物質の動態を可視化
        
        Args:
            nt_levels: 神経伝達物質レベルの時系列データ
            time_points: 時間点
            save_path: 保存パス
        """
        plt.figure(figsize=(12, 6))
        
        for nt_name, levels in nt_levels.items():
            plt.plot(time_points, levels, 
                    label=nt_name,
                    color=self.neurotransmitter_colors[nt_name])
            
        plt.title('神経伝達物質の動態')
        plt.xlabel('時間')
        plt.ylabel('濃度レベル')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_attention_patterns(self,
                              attention_weights: torch.Tensor,
                              head_idx: int = 0,
                              save_path: Optional[str] = None):
        """
        アテンションパターンの可視化
        
        Args:
            attention_weights: アテンション重み [batch_size, num_heads, seq_len, seq_len]
            head_idx: 表示するヘッドのインデックス
            save_path: 保存パス
        """
        # バッチの最初のサンプルのみ使用
        weights = attention_weights[0, head_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, cmap='viridis', annot=True, fmt='.2f')
        plt.title(f'アテンションパターン (Head {head_idx})')
        plt.xlabel('Key位置')
        plt.ylabel('Query位置')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def create_brain_network_visualization(self,
                                        connectivity: torch.Tensor,
                                        node_activities: torch.Tensor,
                                        save_path: Optional[str] = None):
        """
        脳ネットワークの可視化
        
        Args:
            connectivity: 接続行列 [num_nodes, num_nodes]
            node_activities: ノードの活性値 [num_nodes]
            save_path: 保存パス
        """
        # グラフの作成
        G = nx.from_numpy_array(connectivity.cpu().numpy())
        
        # ノードの活性値を正規化
        activities = node_activities.cpu().numpy()
        normalized_activities = (activities - activities.min()) / (activities.max() - activities.min())
        
        # レイアウトの計算
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        plt.figure(figsize=(12, 12))
        
        # エッジの描画
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        # ノードの描画
        node_colors = plt.cm.viridis(normalized_activities)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                             node_size=500, alpha=0.8)
        
        plt.title('脳ネットワーク構造')
        
        # カラーバーの追加
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array([])
        plt.colorbar(sm, label='活性レベル')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def create_3d_brain_activity_animation(self,
                                         activities: torch.Tensor,
                                         save_path: Optional[str] = None):
        """
        3D脳活動のアニメーション
        
        Args:
            activities: 活動データ [time_steps, x, y, z]
            save_path: 保存パス
        """
        fig = go.Figure()
        
        # 時間ステップごとのフレームを作成
        frames = []
        for t in range(activities.shape[0]):
            activity = activities[t].cpu().numpy()
            
            # 3Dボリュームレンダリング
            frame = go.Frame(
                data=[go.Volume(
                    x=np.arange(activity.shape[0]),
                    y=np.arange(activity.shape[1]),
                    z=np.arange(activity.shape[2]),
                    value=activity,
                    opacity=0.1,
                    surface_count=20,
                    colorscale='Viridis'
                )]
            )
            frames.append(frame)
            
        # 初期フレームの設定
        fig.add_trace(frames[0].data[0])
        
        # アニメーションの設定
        fig.frames = frames
        fig.update_layout(
            title='3D脳活動パターン',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                  'fromcurrent': True}]
                }]
            }]
        )
        
        if save_path:
            fig.write_html(save_path)
            
    def plot_neurotransmitter_interactions(self,
                                         interaction_matrix: torch.Tensor,
                                         save_path: Optional[str] = None):
        """
        神経伝達物質間の相互作用を可視化
        
        Args:
            interaction_matrix: 相互作用行列 [num_nt, num_nt]
            save_path: 保存パス
        """
        nt_names = list(self.neurotransmitter_colors.keys())
        matrix = interaction_matrix.cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, 
                   xticklabels=nt_names,
                   yticklabels=nt_names,
                   cmap='RdBu_r',
                   center=0,
                   annot=True,
                   fmt='.2f')
        
        plt.title('神経伝達物質間の相互作用')
        plt.xlabel('Target')
        plt.ylabel('Source')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def create_multi_scale_attention_visualization(self,
                                                attention_weights: List[torch.Tensor],
                                                scale_names: List[str],
                                                save_path: Optional[str] = None):
        """
        マルチスケールアテンションの可視化
        
        Args:
            attention_weights: 各スケールのアテンション重み
            scale_names: スケールの名前
            save_path: 保存パス
        """
        n_scales = len(attention_weights)
        fig = make_subplots(rows=1, cols=n_scales,
                           subplot_titles=scale_names)
        
        for i, weights in enumerate(attention_weights):
            # バッチの最初のサンプルの最初のヘッドを使用
            w = weights[0, 0].cpu().numpy()
            
            fig.add_trace(
                go.Heatmap(z=w,
                          colorscale='Viridis',
                          showscale=(i == n_scales-1)),
                row=1, col=i+1
            )
            
        fig.update_layout(
            title='マルチスケールアテンションパターン',
            height=400,
            width=200 * n_scales
        )
        
        if save_path:
            fig.write_html(save_path)
            
    def plot_learning_dynamics(self,
                             metrics: Dict[str, List[float]],
                             neurotransmitter_levels: Dict[str, List[float]],
                             save_path: Optional[str] = None):
        """
        学習ダイナミクスと神経伝達物質レベルの関係を可視化
        
        Args:
            metrics: 学習指標の時系列データ
            neurotransmitter_levels: 神経伝達物質レベルの時系列データ
            save_path: 保存パス
        """
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('学習指標', '神経伝達物質レベル'))
        
        # 学習指標のプロット
        for metric_name, values in metrics.items():
            fig.add_trace(
                go.Scatter(y=values, name=metric_name),
                row=1, col=1
            )
            
        # 神経伝達物質レベルのプロット
        for nt_name, levels in neurotransmitter_levels.items():
            fig.add_trace(
                go.Scatter(y=levels, name=nt_name,
                          line=dict(color=self.neurotransmitter_colors[nt_name])),
                row=2, col=1
            )
            
        fig.update_layout(height=800, showlegend=True)
        
        if save_path:
            fig.write_html(save_path) 