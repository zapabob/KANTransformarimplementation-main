"""
BioKAN説明可能AIモジュール
モデルの意思決定プロセスを神経科学的視点から解釈・可視化するためのツール
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import pandas as pd
from collections import defaultdict


class BioKANExplainer:
    """
    BioKANモデルの意思決定を説明するための基本クラス
    モデルの内部状態を神経科学的視点から分析・可視化
    """
    
    def __init__(self, model: 'BioKANModel', layer_names: Optional[List[str]] = None):
        """
        初期化
        
        Args:
            model: 説明対象のBioKANモデル
            layer_names: 分析対象のレイヤー名（指定なしの場合はすべて対象）
        """
        self.model = model
        
        # 対象レイヤーの選択
        self.layer_names = layer_names or self.get_default_layers()
        
        # フックとキャッシュの設定
        self.activation_cache = {}
        self.hooks = []
        self._register_hooks()
        
        # 神経伝達物質システム状態のキャッシュ
        self.neuromodulator_states = []
        
        # アストロサイト状態のキャッシュ
        self.astrocyte_states = []
    
    def get_default_layers(self) -> List[str]:
        """
        デフォルトの分析対象レイヤーを取得
        """
        layers = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'biokan_block' in name.lower() or 'classifier' in name.lower():
                layers.append(name)
        return layers
    
    def _register_hooks(self):
        """レイヤーへのフックを登録"""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._create_hook_fn(name))
                self.hooks.append(hook)
    
    def _create_hook_fn(self, name: str) -> Callable:
        """
        フック関数を生成
        
        Args:
            name: レイヤー名
        """
        def hook_fn(module, input, output):
            self.activation_cache[name] = {
                'input': [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input],
                'output': output.detach().clone() if isinstance(output, torch.Tensor) else output
            }
            
            # 神経調節状態の取得（可能な場合）
            if hasattr(module, 'neuromodulator') and module.neuromodulator is not None:
                self.neuromodulator_states.append({
                    'layer': name,
                    'state': module.neuromodulator.get_state()
                })
            
            # アストロサイト状態の取得（可能な場合）
            if hasattr(module, 'astrocyte') and module.astrocyte is not None:
                self.astrocyte_states.append({
                    'layer': name,
                    'effects': module.astrocyte.get_modulatory_effect()
                })
                
        return hook_fn
    
    def _clear_cache(self):
        """キャッシュをクリア"""
        self.activation_cache = {}
        self.neuromodulator_states = []
        self.astrocyte_states = []
    
    def _remove_hooks(self):
        """登録したフックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        モデルの分析を実行
        
        Args:
            inputs: 入力データ
            labels: 正解ラベル（オプション）
            
        Returns:
            分析結果
        """
        # キャッシュのクリア
        self._clear_cache()
        
        # 評価モードでの推論
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # 予測結果の取得
        if outputs.dim() > 1 and outputs.size(1) > 1:  # 分類タスクの場合
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
        else:  # 回帰または二値分類
            probs = torch.sigmoid(outputs) if outputs.dim() == 1 else outputs
            preds = (probs > 0.5).long() if probs.dim() == 1 else probs
        
        # 基本的な分析結果
        analysis = {
            'predictions': preds.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'layer_activations': self._analyze_activations(),
            'neuromodulator_analysis': self._analyze_neuromodulators(),
            'attention_patterns': self._analyze_attention_patterns(),
        }
        
        # 正解ラベルがある場合、正解率も計算
        if labels is not None:
            correct = (preds == labels).float().mean().item()
            analysis['accuracy'] = correct
        
        return analysis
    
    def _analyze_activations(self) -> Dict[str, Any]:
        """レイヤーの活性化状態を分析"""
        activation_stats = {}
        
        for name, cache in self.activation_cache.items():
            # 出力の統計情報
            if isinstance(cache['output'], torch.Tensor):
                output = cache['output']
                activation_stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'sparsity': (output == 0).float().mean().item(),
                    'positive_ratio': (output > 0).float().mean().item(),
                    'negative_ratio': (output < 0).float().mean().item(),
                }
        
        return activation_stats
    
    def _analyze_neuromodulators(self) -> Dict[str, Any]:
        """神経伝達物質の状態を分析"""
        if not self.neuromodulator_states:
            return {}
        
        # 神経伝達物質ごとの統計
        neuromodulator_stats = defaultdict(list)
        
        for state_info in self.neuromodulator_states:
            layer = state_info['layer']
            state = state_info['state']
            
            for nt, value in state.items():
                neuromodulator_stats[nt].append({
                    'layer': layer,
                    'value': value
                })
        
        # 集計
        summary = {}
        for nt, values in neuromodulator_stats.items():
            nt_values = [item['value'] for item in values]
            summary[nt] = {
                'mean': np.mean(nt_values),
                'max': np.max(nt_values),
                'min': np.min(nt_values),
                'std': np.std(nt_values),
                'values_by_layer': values
            }
            
        return summary
    
    def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """アテンションパターンの分析"""
        attention_patterns = {}
        
        for name, cache in self.activation_cache.items():
            # アテンションレイヤーの検出
            if 'attention' in name.lower() and isinstance(cache['output'], torch.Tensor):
                # アテンションの出力パターンを分析
                output = cache['output']
                
                # アテンション重みの抽出（実装によって異なる場合あり）
                # 通常、アテンションモジュールには attention_weights または attn_weights という属性がある
                module = dict(self.model.named_modules())[name]
                attn_weights = None
                
                # 異なる実装方法に対応
                for attr in ['attention_weights', 'attn_weights', 'attn_output_weights']:
                    if hasattr(module, attr):
                        attn_weights = getattr(module, attr)
                        break
                
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    attention_patterns[name] = {
                        'weights': attn_weights.detach().cpu().numpy(),
                        'entropy': self._calculate_attention_entropy(attn_weights),
                        'concentration': self._calculate_attention_concentration(attn_weights),
                    }
        
        return attention_patterns
    
    def _calculate_attention_entropy(self, attn_weights: torch.Tensor) -> np.ndarray:
        """
        アテンション重みのエントロピーを計算
        
        Args:
            attn_weights: アテンション重み（形状: [batch_size, heads, seq_len, seq_len]）
            
        Returns:
            エントロピー値（ヘッドごと）
        """
        # 最後の次元に対してソフトマックスを適用（まだされていない場合）
        if attn_weights.dim() == 4:
            weights = torch.softmax(attn_weights, dim=-1)
            # エントロピー: H = -sum(p * log(p))
            # 非常に小さい値を加えて、log(0)を防ぐ
            epsilon = 1e-12
            entropy = -torch.sum(weights * torch.log(weights + epsilon), dim=-1)
            # ヘッドごとのエントロピー平均
            return entropy.mean(dim=-1).cpu().numpy()
        
        return np.array([])
    
    def _calculate_attention_concentration(self, attn_weights: torch.Tensor) -> np.ndarray:
        """
        アテンション重みの集中度を計算（ジニ係数に似た指標）
        
        Args:
            attn_weights: アテンション重み
            
        Returns:
            集中度（ヘッドごと）
        """
        if attn_weights.dim() == 4:
            weights = torch.softmax(attn_weights, dim=-1)
            batch_size, num_heads, seq_len, _ = weights.shape
            
            # 重みを小さい順にソート
            sorted_weights, _ = torch.sort(weights, dim=-1)
            
            # 累積合計を計算
            cumsum = torch.cumsum(sorted_weights, dim=-1)
            
            # 各位置での理想的な累積合計（完全均等分布）
            indices = torch.arange(1, seq_len + 1, device=weights.device, dtype=torch.float32)
            ideal_cumsum = indices / seq_len
            
            # ジニ係数に似た集中度指標の計算
            # 1に近いほど集中、0に近いほど分散
            concentration = torch.abs(cumsum - ideal_cumsum.view(1, 1, 1, -1)).mean(dim=-1)
            
            return concentration.cpu().numpy()
        
        return np.array([])
    
    def create_counterfactual(self, inputs: torch.Tensor, 
                             target_class: int, 
                             neuromodulator_changes: Dict[str, float] = None,
                             max_iterations: int = 100) -> Dict[str, Any]:
        """
        反事実的説明の生成
        入力データに対して神経調節システムの状態を変更し、
        目標クラスに分類されるような変化を探索
        
        Args:
            inputs: 入力データ
            target_class: 目標クラス
            neuromodulator_changes: 変更する神経伝達物質と変化量
            max_iterations: 最大反復回数
            
        Returns:
            反事実的説明の結果
        """
        # デフォルトの神経伝達物質変化量
        if neuromodulator_changes is None:
            neuromodulator_changes = {
                'dopamine': 0.2,
                'serotonin': 0.1,
                'noradrenaline': 0.15,
                'acetylcholine': 0.1
            }
        
        self.model.eval()
        
        # 元の予測を取得
        with torch.no_grad():
            original_output = self.model(inputs)
            
        if original_output.dim() > 1:
            original_probs = torch.softmax(original_output, dim=1)
            original_pred = torch.argmax(original_probs, dim=1)[0].item()
        else:
            original_probs = torch.sigmoid(original_output)
            original_pred = (original_probs > 0.5).long()[0].item()
        
        # 元の予測が既に目標クラスの場合
        if original_pred == target_class:
            return {
                'success': True,
                'iterations': 0,
                'original_prediction': original_pred,
                'original_confidence': original_probs[0, original_pred].item() if original_probs.dim() > 1 else original_probs[0].item(),
                'counterfactual_prediction': original_pred,
                'counterfactual_confidence': original_probs[0, original_pred].item() if original_probs.dim() > 1 else original_probs[0].item(),
                'neuromodulator_state_changes': []
            }
        
        # 神経伝達物質の状態変化を試行
        neuromodulator_state_changes = []
        current_pred = original_pred
        current_output = original_output
        
        for i in range(max_iterations):
            # モデル内の各レイヤーの神経伝達物質状態を変更
            for name, module in self.model.named_modules():
                if hasattr(module, 'neuromodulator') and module.neuromodulator is not None:
                    current_state = module.neuromodulator.get_state()
                    new_state = current_state.copy()
                    
                    # 各神経伝達物質を変更
                    for nt, change in neuromodulator_changes.items():
                        if nt in new_state:
                            new_state[nt] = min(1.0, max(-1.0, new_state[nt] + change))
                    
                    # 状態を更新
                    module.neuromodulator.set_state(new_state)
                    
                    # 変更を記録
                    neuromodulator_state_changes.append({
                        'iteration': i,
                        'layer': name,
                        'original_state': current_state,
                        'new_state': new_state
                    })
            
            # 変更後の予測
            with torch.no_grad():
                current_output = self.model(inputs)
            
            if current_output.dim() > 1:
                current_probs = torch.softmax(current_output, dim=1)
                current_pred = torch.argmax(current_probs, dim=1)[0].item()
            else:
                current_probs = torch.sigmoid(current_output)
                current_pred = (current_probs > 0.5).long()[0].item()
            
            # 目標クラスに到達した場合
            if current_pred == target_class:
                return {
                    'success': True,
                    'iterations': i + 1,
                    'original_prediction': original_pred,
                    'original_confidence': original_probs[0, original_pred].item() if original_probs.dim() > 1 else original_probs[0].item(),
                    'counterfactual_prediction': current_pred,
                    'counterfactual_confidence': current_probs[0, current_pred].item() if current_probs.dim() > 1 else current_probs[0].item(),
                    'neuromodulator_state_changes': neuromodulator_state_changes
                }
        
        # 最大反復回数に達しても目標クラスに到達しなかった場合
        return {
            'success': False,
            'iterations': max_iterations,
            'original_prediction': original_pred,
            'original_confidence': original_probs[0, original_pred].item() if original_probs.dim() > 1 else original_probs[0].item(),
            'counterfactual_prediction': current_pred,
            'counterfactual_confidence': current_probs[0, current_pred].item() if current_probs.dim() > 1 else current_probs[0].item(),
            'neuromodulator_state_changes': neuromodulator_state_changes
        }
    
    def __del__(self):
        """クリーンアップ"""
        self._remove_hooks()


class FeatureAttributionExplainer(BioKANExplainer):
    """
    特徴量帰属（Feature Attribution）による説明
    入力の各部分が最終予測にどのように貢献しているかを分析
    """
    
    def __init__(self, model: 'BioKANModel', layer_names: Optional[List[str]] = None):
        """初期化"""
        super().__init__(model, layer_names)
        self.gradients = {}
        self.gradient_hooks = []
    
    def _register_gradient_hooks(self):
        """勾配フックを登録"""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                # 順伝播のフック
                hook = module.register_forward_hook(self._create_hook_fn(name))
                self.hooks.append(hook)
                
                # 逆伝播のフック
                hook = module.register_backward_hook(self._create_grad_hook_fn(name))
                self.gradient_hooks.append(hook)
    
    def _create_grad_hook_fn(self, name: str) -> Callable:
        """勾配フック関数を生成"""
        def hook_fn(module, grad_input, grad_output):
            self.gradients[name] = {
                'grad_input': [g.detach().clone() if isinstance(g, torch.Tensor) else g for g in grad_input],
                'grad_output': grad_output[0].detach().clone() if isinstance(grad_output[0], torch.Tensor) else grad_output
            }
        return hook_fn
    
    def _remove_hooks(self):
        """すべてのフックを削除"""
        super()._remove_hooks()
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks = []
    
    def compute_feature_attribution(self, inputs: torch.Tensor, target: Union[int, torch.Tensor], 
                                   method: str = 'integrated_gradients',
                                   steps: int = 20) -> Dict[str, Any]:
        """
        特徴量帰属を計算
        
        Args:
            inputs: 入力データ
            target: 対象クラスまたは対象スコア
            method: 帰属計算方法（'integrated_gradients', 'smooth_grad'）
            steps: 積分ステップ数または平滑化サンプル数
            
        Returns:
            特徴量帰属の結果
        """
        # 入力の勾配を有効にする
        inputs.requires_grad = True
        
        # 勾配フックを登録
        self._register_gradient_hooks()
        
        attribution = None
        
        if method == 'integrated_gradients':
            attribution = self._integrated_gradients(inputs, target, steps)
        elif method == 'smooth_grad':
            attribution = self._smooth_grad(inputs, target, steps)
        else:
            raise ValueError(f"不明な帰属計算方法: {method}")
        
        # 後処理
        inputs.requires_grad = False
        
        return attribution
    
    def _integrated_gradients(self, inputs: torch.Tensor, target: Union[int, torch.Tensor], 
                             steps: int = 20) -> Dict[str, Any]:
        """
        積分勾配による特徴量帰属を計算
        
        Args:
            inputs: 入力データ
            target: 対象クラスまたは対象スコア
            steps: 積分ステップ数
            
        Returns:
            積分勾配による特徴量帰属
        """
        baseline = torch.zeros_like(inputs)
        delta = (inputs - baseline) / steps
        
        integrated_gradients = torch.zeros_like(inputs, dtype=torch.float32)
        
        for i in range(steps):
            # 現在のステップでの入力
            current_input = baseline + delta * i
            current_input.requires_grad = True
            
            # 予測
            self.model.eval()
            outputs = self.model(current_input)
            
            # 勾配のクリア
            self.model.zero_grad()
            
            # ターゲットに対する勾配を計算
            if isinstance(target, int):
                if outputs.dim() > 1:  # 多クラス分類
                    score = outputs[0, target]
                else:  # 二値分類または回帰
                    score = outputs[0]
            else:  # ターゲットがテンソルの場合
                score = outputs[0, target[0]] if outputs.dim() > 1 else outputs[0]
            
            score.backward()
            
            # 入力の勾配を積分
            if current_input.grad is not None:
                integrated_gradients += current_input.grad
            
            # 勾配のクリア
            current_input.grad = None
        
        # 平均勾配計算と入力との要素ごとの積
        attribution = (inputs - baseline) * (integrated_gradients / steps)
        
        return {
            'method': 'integrated_gradients',
            'attribution': attribution.detach().cpu(),
            'input': inputs.detach().cpu(),
            'baseline': baseline.detach().cpu(),
            'steps': steps
        }
    
    def _smooth_grad(self, inputs: torch.Tensor, target: Union[int, torch.Tensor], 
                   samples: int = 20, noise_level: float = 0.15) -> Dict[str, Any]:
        """
        SmoothGradによる特徴量帰属を計算
        
        Args:
            inputs: 入力データ
            target: 対象クラスまたは対象スコア
            samples: サンプル数
            noise_level: ノイズレベル（入力の標準偏差に対する比率）
            
        Returns:
            SmoothGradによる特徴量帰属
        """
        # 入力の標準偏差
        stdev = noise_level * (inputs.max() - inputs.min())
        
        total_gradients = torch.zeros_like(inputs, dtype=torch.float32)
        
        for i in range(samples):
            # ノイズを加えた入力
            noisy_input = inputs + torch.randn_like(inputs) * stdev
            noisy_input.requires_grad = True
            
            # 予測
            self.model.eval()
            outputs = self.model(noisy_input)
            
            # 勾配のクリア
            self.model.zero_grad()
            
            # ターゲットに対する勾配を計算
            if isinstance(target, int):
                if outputs.dim() > 1:  # 多クラス分類
                    score = outputs[0, target]
                else:  # 二値分類または回帰
                    score = outputs[0]
            else:  # ターゲットがテンソルの場合
                score = outputs[0, target[0]] if outputs.dim() > 1 else outputs[0]
            
            score.backward()
            
            # 入力の勾配を集計
            if noisy_input.grad is not None:
                total_gradients += noisy_input.grad
            
            # 勾配のクリア
            noisy_input.grad = None
        
        # 平均勾配
        smoothed_gradients = total_gradients / samples
        
        # 入力との要素ごとの積
        attribution = inputs * smoothed_gradients
        
        return {
            'method': 'smooth_grad',
            'attribution': attribution.detach().cpu(),
            'input': inputs.detach().cpu(),
            'noise_level': noise_level,
            'samples': samples
        } 