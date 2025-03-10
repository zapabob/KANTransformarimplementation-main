"""
生物学的な学習アルゴリズムを実装するモジュール
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, List

class BiologicalOptimizer(optim.Optimizer):
    """
    生物学的な学習則を実装した最適化アルゴリズム
    """
    
    def __init__(self, params, lr=0.001, hebbian_rate=0.0001, homeostatic_rate=0.0001):
        """
        Args:
            params: 最適化するパラメータ
            lr: 学習率
            hebbian_rate: ヘブ学習の学習率
            homeostatic_rate: ホメオスタティック学習の学習率
        """
        if not isinstance(params, (list, tuple)) and not hasattr(params, '__iter__'):
            raise TypeError('パラメータはイテラブルである必要があります。model.parameters()を使用してください。')
            
        defaults = dict(
            lr=lr,
            hebbian_rate=hebbian_rate,
            homeostatic_rate=homeostatic_rate
        )
        
        if lr < 0.0:
            raise ValueError('学習率は0以上である必要があります')
        if hebbian_rate < 0.0:
            raise ValueError('ヘブ学習率は0以上である必要があります')
        if homeostatic_rate < 0.0:
            raise ValueError('ホメオスタティック学習率は0以上である必要があります')
            
        super().__init__(params, defaults)
        
        # 活動履歴の初期化
        self.activity_history = {}
        for group in self.param_groups:
            for p in group['params']:
                self.activity_history[p] = []
                
    def hebbian_update(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        ヘブ則による更新
        
        Args:
            param: パラメータ
            grad: 勾配
            
        Returns:
            update: 更新量
        """
        # パラメータと勾配の形状を取得
        param_shape = param.shape
        
        # 活動相関に基づく更新
        if len(param_shape) == 2:  # 重み行列の場合
            out_features, in_features = param_shape
            
            # 入力と出力の活動を計算
            pre_activity = param.abs().mean(dim=0)  # [in_features]
            post_activity = grad.abs().mean(dim=0)[:out_features]  # [out_features]
            
            # 活動の形状を調整
            pre_activity = pre_activity[:in_features]
            
            # 相関行列を直接計算（外積）
            hebbian_term = post_activity.unsqueeze(1) @ pre_activity.unsqueeze(0)
            
            # 重みの形状に合わせる
            hebbian_term = hebbian_term.reshape(param_shape)
            
        else:  # バイアスなどの1次元パラメータの場合
            hebbian_term = grad.abs().mean() * torch.ones_like(param)
        
        return hebbian_term
        
    def homeostatic_update(self, param: torch.Tensor) -> torch.Tensor:
        """
        ホメオスタティック可塑性による更新
        
        Args:
            param: パラメータ
            
        Returns:
            update: 更新量
        """
        # 活動履歴の更新
        if len(self.activity_history[param]) > 1000:
            self.activity_history[param].pop(0)
            
        current_activity = param.abs().mean().item()
        self.activity_history[param].append(current_activity)
        
        # 平均活動レベルの計算
        mean_activity = torch.tensor(self.activity_history[param]).mean()
        
        # 目標活動レベルとの差に基づく調整
        target_activity = 0.1
        activity_diff = target_activity - mean_activity
        
        return activity_diff * torch.ones_like(param)
        
    def apply_neuromodulation(self, param: torch.Tensor, modulation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        神経伝達物質による学習の調節
        
        Args:
            param: パラメータ
            modulation: 神経伝達物質の調節効果
            
        Returns:
            scale: スケーリング係数
        """
        scale = 1.0
        
        if 'dopamine' in modulation:
            # ドーパミンによる報酬学習の調節
            scale *= (1.0 + 0.5 * modulation['dopamine'])
            
        if 'acetylcholine' in modulation:
            # アセチルコリンによる注意の調節
            scale *= (1.0 + 0.3 * modulation['acetylcholine'])
            
        if 'noradrenaline' in modulation:
            # ノルアドレナリンによる可塑性の調節
            scale *= (1.0 + 0.2 * modulation['noradrenaline'])
            
        return scale
        
    @torch.no_grad()
    def step(self, closure=None, modulation: Optional[Dict[str, torch.Tensor]] = None):
        """
        最適化ステップ
        
        Args:
            closure: 損失値を再計算する関数（オプション）
            modulation: 神経伝達物質の調節効果
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # 勾配の取得
                grad = p.grad
                
                # パラメータの形状に合わせてヘブ則による更新を計算
                hebbian_update = self.hebbian_update(p, grad)
                
                # ホメオスタティック可塑性による更新
                homeostatic_update = self.homeostatic_update(p)
                
                # 更新量の計算
                update = (
                    -group['lr'] * grad +  # 勾配降下
                    group['hebbian_rate'] * hebbian_update +  # ヘブ学習
                    group['homeostatic_rate'] * homeostatic_update  # ホメオスタティック学習
                )
                
                # 神経伝達物質による調節
                if modulation is not None:
                    scale = self.apply_neuromodulation(p, modulation)
                    update *= scale
                    
                # パラメータの更新
                p.add_(update)
                
        return loss
        
class RewardModulatedSTDP(nn.Module):
    """
    報酬で調節されるSTDPを実装したクラス
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # シナプス重み
        self.weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        
        # 適格トレース（register_bufferを使用してデバイス管理を自動化）
        self.register_buffer('eligibility_trace', torch.zeros(hidden_dim, hidden_dim))
        
        # STDPパラメータ
        self.learning_rate = 0.001
        self.trace_decay = 0.95
        
    def update_eligibility_trace(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        """
        適格トレースの更新
        
        Args:
            pre: プレシナプス活動 [batch_size, hidden_dim]
            post: ポストシナプス活動 [batch_size, hidden_dim]
        """
        # バッチ平均を計算して1次元テンソルに変換
        pre_mean = pre.mean(dim=0)  # [hidden_dim]
        post_mean = post.mean(dim=0)  # [hidden_dim]
        
        # STDPによる適格トレースの更新（デバイスの一貫性を保証）
        self.eligibility_trace = self.trace_decay * self.eligibility_trace + (
            torch.outer(post_mean, pre_mean)  # 相関項 [hidden_dim, hidden_dim]
        ).to(self.eligibility_trace.device)
        
    def apply_reward(self, reward: float) -> None:
        """
        報酬信号の適用
        
        Args:
            reward: 報酬値
        """
        # 報酬による重み更新
        self.weights.data += self.learning_rate * reward * self.eligibility_trace
        
        # 重みの範囲を制限
        self.weights.data = torch.clamp(self.weights, -1.0, 1.0)
        
    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        reward: float = 0.0,
        modulation: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        順伝播
        
        Args:
            pre: プレシナプス活動
            post: ポストシナプス活動
            reward: 報酬信号
            modulation: 神経伝達物質の調節効果
            
        Returns:
            output: 出力テンソル
        """
        # 適格トレースの更新
        self.update_eligibility_trace(pre, post)
        
        # 報酬の適用
        if reward != 0.0:
            # 神経伝達物質による報酬の調節
            if modulation is not None and 'dopamine' in modulation:
                reward *= (1.0 + 0.5 * modulation['dopamine'])
                
            self.apply_reward(reward)
            
        # 出力の計算
        output = torch.matmul(pre, self.weights)
        
        return output 