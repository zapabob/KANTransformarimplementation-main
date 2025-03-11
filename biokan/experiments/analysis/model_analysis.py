"""
モデル分析用のユーティリティ
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple
import japanize_matplotlib
from sklearn.manifold import TSNE

class BioKANAnalyzer:
    def __init__(self, model, device="cuda"):
        """
        BioKANモデルの分析クラス
        
        Args:
            model: 分析対象のモデル
            device: 使用するデバイス
        """
        self.model = model
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'neuromodulation_states': []
        }
        
    def plot_learning_curves(self, save_path=None):
        """学習曲線のプロット"""
        plt.figure(figsize=(12, 5))
        
        # 損失の推移
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='訓練損失')
        plt.plot(self.history['val_loss'], label='検証損失')
        plt.title('損失の推移')
        plt.xlabel('エポック')
        plt.ylabel('損失')
        plt.legend()
        plt.grid(True)
        
        # 精度の推移
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='訓練精度')
        plt.plot(self.history['val_acc'], label='検証精度')
        plt.title('精度の推移')
        plt.xlabel('エポック')
        plt.ylabel('精度 (%)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def analyze_class_performance(self, test_loader, class_names=None):
        """クラスごとの性能分析"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)  # MNISTの場合は784次元に平坦化
                
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # 混同行列の計算
        cm = confusion_matrix(targets, predictions)
        
        # 混同行列のプロット
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混同行列')
        plt.xlabel('予測クラス')
        plt.ylabel('実際のクラス')
        if class_names:
            plt.xticks(np.arange(len(class_names)) + 0.5, class_names)
            plt.yticks(np.arange(len(class_names)) + 0.5, class_names)
        plt.show()
        
        # クラスごとの精度を計算
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(class_accuracy)), class_accuracy * 100)
        plt.title('クラスごとの精度')
        plt.xlabel('クラス')
        plt.ylabel('精度 (%)')
        if class_names:
            plt.xticks(range(len(class_names)), class_names)
        plt.grid(True)
        plt.show()
        
        return class_accuracy
        
    def analyze_neuromodulation(self, test_loader):
        """神経調節の効果分析"""
        self.model.eval()
        neuromod_states = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                data = data.view(data.size(0), -1)  # MNISTの場合は784次元に平坦化
                
                # 各ブロックの神経調節状態を取得
                states = []
                h = self.model.input_transform(data)
                h = F.relu(h)
                h = self.model.dropout(h)
                
                for block in self.model.blocks:
                    if block.use_neuromodulation:
                        state = block.neuromodulation.get_state()
                        states.append(state)
                    
                    h = block(h)
                
                neuromod_states.append(states)
        
        # 神経調節の可視化
        avg_states = []
        std_states = []  # 標準偏差も計算
        for i in range(len(self.model.blocks)):
            block_states = [batch[i] for batch in neuromod_states if i < len(batch)]
            if block_states:
                avg_state = {
                    k: np.mean([state[k] for state in block_states])
                    for k in block_states[0].keys()
                }
                std_state = {
                    k: np.std([state[k] for state in block_states])
                    for k in block_states[0].keys()
                }
                avg_states.append(avg_state)
                std_states.append(std_state)
        
        # 神経伝達物質レベルの可視化（平均値）
        plt.figure(figsize=(15, 10))
        
        # サブプロット1: 平均値のバープロット
        plt.subplot(2, 1, 1)
        neurotransmitters = list(avg_states[0].keys())
        x = np.arange(len(self.model.blocks))
        width = 0.8 / len(neurotransmitters)
        
        for i, nt in enumerate(neurotransmitters):
            values = [state[nt] for state in avg_states]
            errors = [state[nt] for state in std_states]
            plt.bar(x + i * width, values, width, label=nt, 
                   yerr=errors, capsize=5)
        
        plt.title('ブロックごとの神経伝達物質レベル（平均±標準偏差）')
        plt.xlabel('ブロック番号')
        plt.ylabel('活性化レベル')
        plt.legend(title='神経伝達物質', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # サブプロット2: 時系列プロット
        plt.subplot(2, 1, 2)
        for i, nt in enumerate(neurotransmitters):
            values = [state[nt] for state in avg_states]
            errors = [state[nt] for state in std_states]
            plt.plot(x, values, 'o-', label=nt, linewidth=2)
            plt.fill_between(x, 
                           [v - e for v, e in zip(values, errors)],
                           [v + e for v, e in zip(values, errors)],
                           alpha=0.2)
        
        plt.title('神経伝達物質レベルの変化')
        plt.xlabel('ブロック番号')
        plt.ylabel('活性化レベル')
        plt.legend(title='神経伝達物質', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('neuromodulation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return avg_states, std_states
    
    def analyze_attention_patterns(self, data_loader):
        """アテンションパターンの分析と可視化
        
        Args:
            data_loader: データローダー
            
        Returns:
            patterns: アテンションパターンのリスト
        """
        self.model.eval()
        
        # サンプルデータの取得
        images, _ = next(iter(data_loader))
        images = images.to(self.device)
        images = images.view(images.size(0), -1)  # 入力を平坦化
        
        # アテンションパターンの取得
        with torch.no_grad():
            patterns = []
            h = self.model.input_transform(images)
            h = F.relu(h)
            h = self.model.dropout(h)
            
            for block in self.model.blocks:
                # feature_attentionを使用してアテンション重みを取得
                _, attn_weights = block.feature_attention(h, need_weights=True)
                # ヘッドの平均を取る
                avg_pattern = attn_weights.mean(dim=1).cpu().numpy()
                patterns.append(avg_pattern)
                h = block(h)
        
        # 可視化
        num_layers = len(patterns)
        rows = (num_layers + 2) // 3  # 3列で表示
        cols = min(3, num_layers)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()
        
        for i, pattern in enumerate(patterns):
            # バッチの最初のサンプルのみを表示
            sns.heatmap(pattern[0], cmap='viridis', ax=axes[i])
            axes[i].set_title(f'レイヤー {i+1}のアテンションマップ')
            axes[i].set_xlabel('クエリ位置')
            axes[i].set_ylabel('キー位置')
        
        # 使用していない軸を非表示
        for i in range(len(patterns), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('各レイヤーのアテンションパターン分析', fontsize=14)
        plt.tight_layout()
        plt.savefig('attention_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return patterns

    def visualize_feature_space(self, data_loader, n_samples=1000):
        """特徴空間の可視化（t-SNE使用）"""
        self.model.eval()
        features = []
        labels = []
        count = 0
        
        with torch.no_grad():
            for images, batch_labels in data_loader:
                if count >= n_samples:
                    break
                    
                images = images.to(self.device)
                # モデルの中間層の特徴量を取得
                batch_features = self.model.get_features(images)
                features.append(batch_features.cpu().numpy())
                labels.extend(batch_labels.numpy())
                
                count += len(images)
        
        # 特徴量の結合
        features = np.concatenate(features, axis=0)
        features = features.reshape(features.shape[0], -1)  # 2D形状に変換
        
        # t-SNEによる次元削減
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(features)
        
        # 可視化
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='数字クラス')
        plt.title('t-SNEによる特徴空間の可視化')
        plt.xlabel('第1主成分')
        plt.ylabel('第2主成分')
        
        # クラスごとの凡例を追加
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=plt.cm.tab10(i/10), 
                                    label=f'数字 {i}', markersize=10)
                         for i in range(10)]
        plt.legend(handles=legend_elements, title='クラス', 
                  loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig('feature_space.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return features_2d, labels
    
    def prepare_transfer_learning(self, new_num_classes):
        """転移学習の準備"""
        # 出力層の重みを保存
        old_weights = self.model.output_transform.weight.data.clone()
        old_bias = self.model.output_transform.bias.data.clone()
        
        # 新しい出力層を作成
        in_features = self.model.output_transform.in_features
        self.model.output_transform = torch.nn.Linear(in_features, new_num_classes).to(self.device)
        
        # 共通のクラスに対して重みを転送
        min_classes = min(new_num_classes, old_weights.size(0))
        self.model.output_transform.weight.data[:min_classes] = old_weights[:min_classes]
        self.model.output_transform.bias.data[:min_classes] = old_bias[:min_classes]
        
        # 特徴抽出部分の勾配を無効化
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 出力層の勾配を有効化
        for param in self.model.output_transform.parameters():
            param.requires_grad = True
            
    def visualize_feature_space(self, test_loader, num_samples=1000):
        """特徴空間の可視化"""
        self.model.eval()
        features = []
        labels = []
        count = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                if count >= num_samples:
                    break
                    
                data = data.to(self.device)
                data = data.view(data.size(0), -1)
                
                # 最後の出力層の直前の特徴を取得
                h = self.model.input_transform(data)
                h = F.relu(h)
                h = self.model.dropout(h)
                
                for block in self.model.blocks:
                    h = block(h)
                
                features.append(h.cpu().numpy())
                labels.extend(target.numpy())
                count += data.size(0)
        
        features = np.concatenate(features)
        labels = np.array(labels)
        
        # t-SNEで次元削減
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # 可視化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('t-SNEによる特徴空間の可視化')
        plt.xlabel('第1主成分')
        plt.ylabel('第2主成分')
        plt.show() 