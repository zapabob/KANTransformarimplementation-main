import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BioKANExplainer:
    """
    BioKANモデルの説明可能性を提供するクラス
    - 内部状態の可視化
    - 神経伝達物質の活動分析
    - 注意機構の可視化
    - 反実仮想分析
    """
    def __init__(self, model, device=None):
        """
        初期化
        
        Args:
            model: 説明対象のBioKANモデル
            device: 計算デバイス（Noneでモデルのデバイスをそのまま使用）
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        
        # 内部状態のキャッシュ
        self.cached_states = {}
        self.cached_attention = {}
        
        # 登録されたフックのリスト
        self.hooks = []
        
        # フック関数を登録
        self._register_hooks()
    
    def _register_hooks(self):
        """モデルにフック関数を登録して内部状態をキャプチャ"""
        # フック関数をクリア
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # アテンションレイヤーのフック関数
        def attention_hook(name):
            def hook_fn(module, input, output):
                # output[0]は出力テンソル、output[1]はアテンション重み
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    self.cached_attention[name] = output[1].detach()
            return hook_fn
        
        # 神経調節レイヤーのフック関数
        def neuromod_hook(name):
            def hook_fn(module, input, output):
                if hasattr(module, 'current_state'):
                    self.cached_states[name] = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                              for k, v in module.current_state.items()}
            return hook_fn
        
        # モデル内の全レイヤーをチェックして適切なフックを登録
        for name, module in self.model.named_modules():
            # アテンションモジュールの識別と登録
            if 'attention' in name.lower() or hasattr(module, 'q_proj'):
                hook = module.register_forward_hook(attention_hook(name))
                self.hooks.append(hook)
            
            # 神経調節モジュールの識別と登録
            if ('neuromod' in name.lower() or 
                hasattr(module, 'neuromodulator') or 
                hasattr(module, 'astrocyte')):
                hook = module.register_forward_hook(neuromod_hook(name))
                self.hooks.append(hook)
    
    def explain(self, input_data, target=None, include_counterfactuals=True) -> Dict[str, Any]:
        """
        モデルの内部状態と判断を説明する
        
        Args:
            input_data: 入力データ
            target: 目標クラス（分類モデルの場合）
            include_counterfactuals: 反実仮想分析を含めるかどうか
            
        Returns:
            説明データの辞書
        """
        # キャッシュをクリア
        self.cached_states = {}
        self.cached_attention = {}
        
        # 評価モードに設定
        self.model.eval()
        
        # 入力データの形状確認と調整
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, device=self.device)
        
        if input_data.dim() == 2:
            # バッチ次元を追加 [seq_len, dim] -> [1, seq_len, dim]
            input_data = input_data.unsqueeze(0)
        
        with torch.no_grad():
            # モデルの実行
            outputs = self.model(input_data)
            
            # 出力の形式によって処理を分岐
            if isinstance(outputs, tuple):
                prediction = outputs[0]
            else:
                prediction = outputs
            
            # 予測クラスの取得（分類モデルの場合）
            if prediction.dim() > 1 and prediction.size(-1) > 1:
                pred_class = prediction.argmax(dim=-1)
                prediction_probs = F.softmax(prediction, dim=-1)
            else:
                pred_class = prediction
                prediction_probs = prediction
            
            # 説明データの構築
            explanation = {
                'input': input_data.cpu().numpy(),
                'prediction': prediction.cpu().numpy(),
                'predicted_class': pred_class.cpu().numpy(),
                'attention_patterns': {k: v.cpu().numpy() for k, v in self.cached_attention.items()},
                'layer_states': [v for k, v in sorted(self.cached_states.items())]
            }
            
            # 反実仮想分析（オプション）
            if include_counterfactuals:
                cf_results = self._generate_counterfactuals(input_data)
                explanation['counterfactuals'] = cf_results
        
        return explanation
    
    def _generate_counterfactuals(self, input_data) -> Dict[str, List[Dict[str, Any]]]:
        """
        反実仮想分析を生成（「もし神経伝達物質の状態が異なっていたら？」）
        
        Args:
            input_data: 入力データ
            
        Returns:
            反実仮想分析結果の辞書
        """
        # 重要な神経伝達物質のリスト
        neurotransmitters = ['dopamine', 'serotonin', 'noradrenaline', 'acetylcholine']
        # 変異レベルのリスト
        levels = [-0.8, -0.4, 0.0, 0.4, 0.8]
        
        counterfactuals = {}
        
        # 各神経伝達物質について
        for nt in neurotransmitters:
            cf_results = []
            
            for level in levels:
                # 神経伝達物質状態を設定
                if hasattr(self.model, 'set_neuromodulator_state'):
                    old_state = self.model.get_neuromodulator_state()
                    new_state = old_state.copy()
                    new_state[nt] = level
                    self.model.set_neuromodulator_state(new_state)
                
                # 前方伝播を実行
                with torch.no_grad():
                    outputs = self.model(input_data)
                    
                    if isinstance(outputs, tuple):
                        prediction = outputs[0]
                    else:
                        prediction = outputs
                    
                    if prediction.dim() > 1 and prediction.size(-1) > 1:
                        pred_class = prediction.argmax(dim=-1)
                        prediction_probs = F.softmax(prediction, dim=-1)
                    else:
                        pred_class = prediction
                        prediction_probs = prediction
                
                # 結果を保存
                cf_result = {
                    'neuromodulator': {nt: level},
                    'prediction': prediction.cpu().numpy(),
                    'predicted_class': pred_class.cpu().numpy()
                }
                cf_results.append(cf_result)
                
                # 元の状態に戻す
                if hasattr(self.model, 'set_neuromodulator_state'):
                    self.model.set_neuromodulator_state(old_state)
            
            counterfactuals[nt] = cf_results
        
        return counterfactuals
    
    def generate_attention_heatmap(self, layer_name=None, head_idx=None, output_path=None):
        """
        アテンション重みをヒートマップで可視化
        
        Args:
            layer_name: 可視化するレイヤー名（Noneですべて）
            head_idx: 可視化するヘッドインデックス（Noneで平均）
            output_path: 出力ファイルパス
            
        Returns:
            生成された図のリスト
        """
        figures = []
        
        if layer_name is not None:
            # 特定のレイヤーのみを可視化
            if layer_name in self.cached_attention:
                attn = self.cached_attention[layer_name]
                fig = self._plot_attention_heatmap(attn, layer_name, head_idx)
                figures.append(fig)
                
                if output_path:
                    if head_idx is not None:
                        save_path = f"{output_path}_{layer_name}_head{head_idx}.png"
                    else:
                        save_path = f"{output_path}_{layer_name}.png"
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            # すべてのレイヤーを可視化
            for name, attn in self.cached_attention.items():
                fig = self._plot_attention_heatmap(attn, name, head_idx)
                figures.append(fig)
                
                if output_path:
                    if head_idx is not None:
                        save_path = f"{output_path}_{name}_head{head_idx}.png"
                    else:
                        save_path = f"{output_path}_{name}.png"
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return figures
    
    def _plot_attention_heatmap(self, attention_weights, layer_name, head_idx=None):
        """アテンション重みのヒートマップをプロット"""
        # NumPy配列に変換
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # 4次元: [batch_size, num_heads, seq_len, seq_len]
        # 3次元: [num_heads, seq_len, seq_len]
        # 2次元: [seq_len, seq_len]
        if attention_weights.ndim == 4:
            # バッチの最初のサンプルを使用
            attention_weights = attention_weights[0]
        
        if head_idx is not None and attention_weights.ndim > 2:
            # 特定のヘッドを選択
            attention_map = attention_weights[head_idx]
            title = f"{layer_name} - Head {head_idx}"
        elif attention_weights.ndim > 2:
            # すべてのヘッドの平均
            attention_map = attention_weights.mean(axis=0)
            title = f"{layer_name} - Average Heads"
        else:
            attention_map = attention_weights
            title = f"{layer_name}"
        
        # ヒートマップの作成
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_map, cmap="viridis", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        
        return fig
    
    def visualize_neuromodulation(self, output_path=None):
        """
        神経調節状態を可視化
        
        Args:
            output_path: 出力ファイルパス
            
        Returns:
            生成された図
        """
        fig, axes = plt.subplots(1, len(self.cached_states), figsize=(15, 5))
        
        if len(self.cached_states) == 1:
            axes = [axes]
        
        for i, (name, state) in enumerate(sorted(self.cached_states.items())):
            ax = axes[i]
            
            # 神経伝達物質のリストを抽出
            if 'neuromodulator' in state:
                neuromod = state['neuromodulator']
                
                # バーチャートで神経伝達物質レベルを表示
                if isinstance(neuromod, dict):
                    nt_names = list(neuromod.keys())
                    nt_values = [neuromod[n] for n in nt_names]
                    
                    ax.bar(nt_names, nt_values)
                    ax.set_ylim(-1, 1)
                    ax.set_title(f"{name} - Neuromodulators")
                    ax.tick_params(axis='x', rotation=45)
            
            # グリア細胞のアクティビティを表示
            elif 'astrocyte' in state or 'microglia' in state:
                if 'astrocyte' in state:
                    astro = state['astrocyte']
                    if isinstance(astro, dict):
                        astro_keys = list(astro.keys())
                        astro_values = [astro[k] for k in astro_keys]
                        ax.bar(astro_keys, astro_values, color='skyblue')
                        ax.set_title(f"{name} - Astrocyte")
                        ax.tick_params(axis='x', rotation=45)
                
                if 'microglia' in state:
                    micro = state['microglia']
                    if isinstance(micro, dict):
                        micro_keys = list(micro.keys())
                        micro_values = [micro[k] for k in micro_keys]
                        ax.bar(micro_keys, micro_values, color='salmon')
                        ax.set_title(f"{name} - Microglia")
                        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_counterfactual_analysis(self, target_nt=None, output_path=None):
        """
        反実仮想分析の結果をプロット
        
        Args:
            target_nt: 対象の神経伝達物質（NoneですべてのNTをサブプロット）
            output_path: 出力ファイルパス
            
        Returns:
            生成された図
        """
        if not hasattr(self, 'cached_counterfactuals'):
            raise ValueError("No counterfactual results available. Run explain() with include_counterfactuals=True first.")
        
        counterfactuals = self.cached_counterfactuals
        
        if target_nt is not None:
            # 特定の神経伝達物質の結果をプロット
            if target_nt not in counterfactuals:
                raise ValueError(f"Target neurotransmitter '{target_nt}' not found in counterfactual results.")
            
            fig = self._plot_single_counterfactual(counterfactuals[target_nt], target_nt)
            
            if output_path:
                plt.savefig(f"{output_path}_{target_nt}.png", dpi=300, bbox_inches='tight')
            
            return fig
        else:
            # すべての神経伝達物質の結果をサブプロットに
            nts = list(counterfactuals.keys())
            fig, axes = plt.subplots(len(nts), 1, figsize=(10, 4 * len(nts)))
            
            if len(nts) == 1:
                axes = [axes]
            
            for i, nt in enumerate(nts):
                self._plot_single_counterfactual(counterfactuals[nt], nt, ax=axes[i])
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    def _plot_single_counterfactual(self, cf_results, nt_name, ax=None):
        """単一の神経伝達物質に対する反実仮想分析をプロット"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = plt.gcf()
        
        # レベルと予測値を抽出
        levels = [r['neuromodulator'][nt_name] for r in cf_results]
        
        # 予測値のタイプに応じた処理
        if cf_results[0]['prediction'].ndim > 1:
            # 分類問題の場合：各クラスの確率をプロット
            num_classes = cf_results[0]['prediction'].shape[-1]
            for cls in range(num_classes):
                probs = [r['prediction'][0, cls] for r in cf_results]
                ax.plot(levels, probs, 'o-', label=f'Class {cls}')
            
            ax.set_ylabel('Prediction Probability')
            ax.legend()
        else:
            # 回帰問題の場合：予測値をプロット
            predictions = [r['prediction'][0] for r in cf_results]
            ax.plot(levels, predictions, 'o-')
            ax.set_ylabel('Prediction Value')
        
        ax.set_title(f'Effect of {nt_name} on Prediction')
        ax.set_xlabel(f'{nt_name} Level')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def __del__(self):
        """デストラクタ：すべてのフックを削除"""
        for hook in self.hooks:
            hook.remove()


class FeatureAttributionExplainer:
    """
    特徴量帰属分析のためのエクスプレイナー
    - 勾配ベースの特徴量重要度分析
    - 統合勾配法
    - 摂動感度分析
    """
    def __init__(self, model, device=None):
        """
        初期化
        
        Args:
            model: 説明対象のモデル
            device: 計算デバイス
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
    
    def compute_gradients(self, input_data, target_class=None):
        """
        入力に対する勾配を計算
        
        Args:
            input_data: 入力データ
            target_class: 目標クラス（分類モデルの場合）
            
        Returns:
            入力に対する勾配
        """
        # 入力データをテンソルに変換
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, device=self.device)
        
        # バッチ次元を追加
        if input_data.dim() == 2:
            input_data = input_data.unsqueeze(0)
        
        # 勾配を計算するために入力に対して微分を有効化
        input_data.requires_grad_(True)
        
        # モデルの出力を計算
        output = self.model(input_data)
        
        # 出力が複数ある場合は最初の要素を使用
        if isinstance(output, tuple):
            output = output[0]
        
        # 目標クラスが指定されていなければ予測クラスを使用
        if target_class is None:
            if output.dim() > 1 and output.size(-1) > 1:
                target_class = output.argmax(dim=-1)
            else:
                target = output
        
        # スカラー値を取得するためのインデックス付け
        if output.dim() > 1 and output.size(-1) > 1:
            scores = output[:, target_class]
        else:
            scores = output
        
        # 勾配の計算
        self.model.zero_grad()
        scores.backward(torch.ones_like(scores))
        
        # 勾配を取得
        gradients = input_data.grad.detach()
        
        return gradients
    
    def integrated_gradients(self, input_data, target_class=None, steps=50, baseline=None):
        """
        統合勾配法による特徴量帰属
        
        Args:
            input_data: 入力データ
            target_class: 目標クラス
            steps: 積分ステップ数
            baseline: ベースライン入力（Noneでゼロベクトル）
            
        Returns:
            特徴量帰属スコア
        """
        # 入力データをテンソルに変換
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, device=self.device)
        
        # バッチ次元を追加
        if input_data.dim() == 2:
            input_data = input_data.unsqueeze(0)
        
        # ベースラインを設定
        if baseline is None:
            baseline = torch.zeros_like(input_data)
        elif not isinstance(baseline, torch.Tensor):
            baseline = torch.tensor(baseline, device=self.device)
            if baseline.dim() == 2:
                baseline = baseline.unsqueeze(0)
        
        # パスの作成（ベースラインから入力まで）
        alphas = torch.linspace(0, 1, steps, device=self.device)
        path_inputs = [baseline + alpha * (input_data - baseline) for alpha in alphas]
        
        # 各パスポイントでの勾配を計算
        gradients = []
        for path_input in path_inputs:
            path_input = path_input.detach().requires_grad_(True)
            grad = self.compute_gradients(path_input, target_class)
            gradients.append(grad)
        
        # 勾配を積分
        integrated_grads = torch.zeros_like(input_data)
        for grad in gradients:
            integrated_grads += grad
        
        # 平均を取り、入力-ベースラインとの積を計算
        integrated_grads = integrated_grads / len(gradients)
        attributions = integrated_grads * (input_data - baseline)
        
        return attributions
    
    def occlusion_sensitivity(self, input_data, target_class=None, window_size=1, occlusion_value=0):
        """
        遮蔽感度分析による特徴量帰属
        
        Args:
            input_data: 入力データ
            target_class: 目標クラス
            window_size: 遮蔽窓サイズ
            occlusion_value: 遮蔽値
            
        Returns:
            特徴量帰属スコア
        """
        # 入力データをテンソルに変換
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, device=self.device)
        
        # バッチ次元を追加
        if input_data.dim() == 2:
            input_data = input_data.unsqueeze(0)
        
        # 元の予測を計算
        with torch.no_grad():
            original_output = self.model(input_data)
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            
            # 目標クラスが指定されていなければ予測クラスを使用
            if target_class is None:
                if original_output.dim() > 1 and original_output.size(-1) > 1:
                    target_class = original_output.argmax(dim=-1)
                else:
                    target_class = None
            
            # スカラー値を取得するためのインデックス付け
            if original_output.dim() > 1 and original_output.size(-1) > 1:
                original_score = original_output[:, target_class].item()
            else:
                original_score = original_output.item()
        
        # 帰属スコアの初期化
        attributions = torch.zeros_like(input_data)
        
        # 各特徴量を遮蔽して効果を計測
        for i in range(input_data.size(1)):
            for j in range(input_data.size(2)):
                # 遮蔽範囲
                start_i = max(0, i - window_size // 2)
                end_i = min(input_data.size(1), i + window_size // 2 + 1)
                start_j = max(0, j - window_size // 2)
                end_j = min(input_data.size(2), j + window_size // 2 + 1)
                
                # 入力のコピーを作成
                occluded_input = input_data.clone()
                occluded_input[:, start_i:end_i, start_j:end_j] = occlusion_value
                
                # 遮蔽した入力での予測
                with torch.no_grad():
                    occluded_output = self.model(occluded_input)
                    if isinstance(occluded_output, tuple):
                        occluded_output = occluded_output[0]
                    
                    # スカラー値を取得
                    if occluded_output.dim() > 1 and occluded_output.size(-1) > 1:
                        occluded_score = occluded_output[:, target_class].item()
                    else:
                        occluded_score = occluded_output.item()
                
                # 帰属スコアの計算（元の予測 - 遮蔽後の予測）
                attributions[0, i, j] = original_score - occluded_score
        
        return attributions
    
    def plot_feature_attributions(self, input_data, attributions, output_path=None):
        """
        特徴量帰属をヒートマップで可視化
        
        Args:
            input_data: 元の入力データ
            attributions: 特徴量帰属スコア
            output_path: 出力ファイルパス
            
        Returns:
            生成された図
        """
        # テンソルをNumPy配列に変換
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.cpu().numpy()
        
        # バッチ次元を除去
        if input_data.ndim > 2:
            input_data = input_data[0]
        if attributions.ndim > 2:
            attributions = attributions[0]
        
        # 可視化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 元の入力
        im0 = axes[0].imshow(input_data, cmap='viridis')
        axes[0].set_title('Original Input')
        plt.colorbar(im0, ax=axes[0])
        
        # 特徴量帰属
        im1 = axes[1].imshow(attributions, cmap='coolwarm', center=0)
        axes[1].set_title('Feature Attributions')
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig 