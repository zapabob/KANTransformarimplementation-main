"""
グリア細胞（アストロサイト、ミクログリアなど）のシミュレーション
神経回路の調節と恒常性維持の機能を実装
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class Astrocyte:
    """
    アストロサイト（星状膠細胞）
    神経伝達物質のリサイクル、シナプス調節、エネルギー供給などの機能を担当
    """
    
    def __init__(self, region_shape: Tuple[int, ...], activation_threshold: float = 0.7, 
                 decay_rate: float = 0.05, diffusion_rate: float = 0.1,
                 temporal_integration_capacity: float = 0.8,
                 layer_coupling_strength: float = 0.6):
        """
        初期化
        
        Args:
            region_shape: 担当する脳領域の形状 (例: (10, 10) で2D領域)
            activation_threshold: 活性化閾値
            decay_rate: 活性の減衰率
            diffusion_rate: Ca²⁺波の拡散率
            temporal_integration_capacity: 時間的統合能力（0〜1）
            layer_coupling_strength: 層間結合強度（0〜1）
        """
        self.region_shape = region_shape
        self.activation_threshold = activation_threshold
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate
        self.temporal_integration_capacity = temporal_integration_capacity
        self.layer_coupling_strength = layer_coupling_strength
        
        # 活性化状態 (カルシウムウェーブの強度を表す)
        self.activation = np.zeros(region_shape)
        
        # グルタミン酸・GABA濃度
        self.glutamate_level = np.zeros(region_shape)
        self.gaba_level = np.zeros(region_shape)
        
        # 神経栄養因子の放出レベル
        self.trophic_factors = np.zeros(region_shape)
        
        # 皮質層間時間差処理のための変数
        self.temporal_delay_buffer = []  # 過去の活性状態を保存するバッファ
        self.buffer_max_size = 5  # 最大バッファサイズ（時間窓の長さ）
        self.layer_specific_delays = np.linspace(0.1, 1.0, 6)  # 各層（6層）の特有の時間遅延
        self.cross_layer_modulation = np.zeros(region_shape)  # 層間変調効果
    
    def update(self, neural_activity: np.ndarray, layer_index: int = 0, delta_t: float = 1.0):
        """
        アストロサイトの状態を更新
        
        Args:
            neural_activity: 神経活動のマップ
            layer_index: 皮質層のインデックス（0〜5）
            delta_t: 時間ステップ
        """
        # 神経活動に基づく活性化
        activation_input = np.where(neural_activity > self.activation_threshold, 
                                    neural_activity, 0)
        self.activation += activation_input * delta_t
        
        # カルシウムウェーブの拡散（簡易的な拡散モデル）
        # 畳み込みフィルタを使った拡散
        kernel = np.array([[0.05, 0.1, 0.05], 
                           [0.1, 0.4, 0.1], 
                           [0.05, 0.1, 0.05]])
        
        padded = np.pad(self.activation, 1, mode='constant')
        diffused = np.zeros_like(self.activation)
        
        # 2D拡散の場合
        if len(self.region_shape) == 2:
            for i in range(self.region_shape[0]):
                for j in range(self.region_shape[1]):
                    diffused[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        
        # 拡散と減衰
        self.activation = (1 - self.diffusion_rate) * self.activation + self.diffusion_rate * diffused
        self.activation *= (1 - self.decay_rate * delta_t)
        
        # グルタミン酸・GABAレベルの更新
        self.glutamate_level = np.maximum(0, neural_activity * 0.8 - self.activation * 0.4)
        self.gaba_level = np.maximum(0, -neural_activity * 0.6 + self.activation * 0.3)
        
        # 栄養因子の更新
        self.trophic_factors = self.activation * 0.5 * np.maximum(0, 1 - np.abs(neural_activity))
        
        # 皮質層間時間差処理
        self._process_temporal_cortical_layers(neural_activity, layer_index, delta_t)
    
    def _process_temporal_cortical_layers(self, current_activity: np.ndarray, layer_index: int, delta_t: float):
        """
        皮質層間の時間差を処理
        
        Args:
            current_activity: 現在の層の神経活動
            layer_index: 皮質層のインデックス
            delta_t: 時間ステップ
        """
        # 現在の活動を時間バッファに追加
        layer_activity_info = {
            'activity': current_activity.copy(),
            'layer': layer_index,
            'time': delta_t * self.layer_specific_delays[layer_index]  # 層に特有の時間遅延
        }
        
        self.temporal_delay_buffer.append(layer_activity_info)
        
        # バッファサイズを制限
        if len(self.temporal_delay_buffer) > self.buffer_max_size:
            self.temporal_delay_buffer.pop(0)
        
        # 層間の時間差に基づく調節計算
        self.cross_layer_modulation = np.zeros_like(self.activation)
        
        if len(self.temporal_delay_buffer) > 1:
            # 現在の層と過去の層との活動の時間差を分析
            current_layer = layer_index
            
            for past_info in self.temporal_delay_buffer[:-1]:  # 最新の要素以外を処理
                past_layer = past_info['layer']
                past_activity = past_info['activity']
                time_diff = past_info['time']
                
                # 層間の時間差重み付け（皮質層の階層関係をモデル化）
                # 例: 層6→層4の信号伝達は遅い、層4→層2/3は早いなど
                layer_distance = abs(current_layer - past_layer)
                temporal_weight = np.exp(-layer_distance * time_diff)
                
                # トップダウン（高次→低次）とボトムアップ（低次→高次）の方向性を区別
                if current_layer < past_layer:  # トップダウン
                    direction_factor = 0.7  # トップダウン効果は抑制的に
                else:  # ボトムアップ
                    direction_factor = 1.3  # ボトムアップ効果は促進的に
                
                # 層間の時間差変調を計算
                layer_modulation = (current_activity - past_activity) * temporal_weight * direction_factor
                
                # 変調効果の蓄積（時間的統合）
                self.cross_layer_modulation += layer_modulation * self.temporal_integration_capacity
        
        # 変調効果を正規化
        if np.max(np.abs(self.cross_layer_modulation)) > 0:
            self.cross_layer_modulation /= np.max(np.abs(self.cross_layer_modulation))
        
        # 層間カップリング強度で変調効果をスケーリング
        self.cross_layer_modulation *= self.layer_coupling_strength
    
    def get_modulatory_effect(self) -> Dict[str, np.ndarray]:
        """
        アストロサイトの調節効果を取得
        
        Returns:
            調節効果のマップ
        """
        return {
            'glutamate_uptake': 1.0 - 0.8 * self.glutamate_level,  # グルタミン酸取り込み
            'gaba_uptake': 1.0 - 0.7 * self.gaba_level,            # GABA取り込み
            'synapse_modulation': 0.8 + 0.4 * self.activation,     # シナプス強度の調節
            'trophic_support': self.trophic_factors,               # 栄養支援
            'cross_layer_temporal_modulation': self.cross_layer_modulation  # 皮質層間時間差調節
        }


class Microglia:
    """
    ミクログリア（小膠細胞）
    免疫防御、シナプス刈り込み、炎症反応などを担当
    """
    
    def __init__(self, region_shape: Tuple[int, ...], activation_threshold: float = 0.8, 
                 mobility: float = 0.2, pruning_threshold: float = 0.3):
        """
        初期化
        
        Args:
            region_shape: 担当する脳領域の形状
            activation_threshold: 活性化閾値
            mobility: 移動性（0～1）
            pruning_threshold: シナプス刈り込みの閾値
        """
        self.region_shape = region_shape
        self.activation_threshold = activation_threshold
        self.mobility = mobility
        self.pruning_threshold = pruning_threshold
        
        # 活性化状態
        self.activation = np.zeros(region_shape)
        
        # 位置（密度分布）
        self.density = np.ones(region_shape) * 0.5
        
        # 炎症状態
        self.inflammation = np.zeros(region_shape)
        
        # 刈り込み活性
        self.pruning_activity = np.zeros(region_shape)
    
    def update(self, damage_signals: np.ndarray, weak_synapses: np.ndarray, delta_t: float = 1.0):
        """
        ミクログリアの状態を更新
        
        Args:
            damage_signals: 損傷シグナルのマップ（異常な活動や細胞損傷を示す）
            weak_synapses: 弱いシナプスのマップ（刈り込みの対象）
            delta_t: 時間ステップ
        """
        # 損傷シグナルに応じた活性化
        activation_input = np.where(damage_signals > self.activation_threshold, 
                                    damage_signals, 0)
        self.activation += activation_input * delta_t
        
        # 自然減衰
        self.activation *= 0.95
        
        # 炎症反応の更新
        self.inflammation = self.activation * 0.8
        
        # 密度分布の更新（損傷シグナルに向かって移動）
        target_density = np.zeros_like(self.density)
        # 損傷シグナルの強い場所に高い密度を設定
        target_density = 0.2 + 0.8 * damage_signals
        
        # 現在の密度から目標密度へ徐々に移動
        self.density = (1 - self.mobility * delta_t) * self.density + self.mobility * delta_t * target_density
        
        # 刈り込み活性の更新
        pruning_target = np.where(weak_synapses < self.pruning_threshold, 
                                 1.0, 0.0)
        self.pruning_activity = self.density * pruning_target
    
    def get_modulatory_effect(self) -> Dict[str, np.ndarray]:
        """
        ミクログリアの調節効果を取得
        
        Returns:
            調節効果のマップ
        """
        return {
            'synapse_pruning': self.pruning_activity,               # シナプス刈り込み
            'inflammatory_response': self.inflammation,             # 炎症反応
            'neuroprotection': 0.8 * self.activation * (1 - self.inflammation)  # 神経保護
        }


class PharmacologicalModulator:
    """
    薬理学的調節因子
    薬物や化学物質による神経系への影響をシミュレート
    """
    
    def __init__(self):
        """初期化"""
        # 登録された薬物とその効果
        self.registered_drugs = {
            # ドーパミン系
            'l-dopa': {'dopamine': 0.8, 'duration': 20},
            'haloperidol': {'dopamine': -0.7, 'duration': 30},
            
            # セロトニン系
            'ssri': {'serotonin': 0.6, 'duration': 50},
            'mdma': {'serotonin': 0.9, 'dopamine': 0.5, 'noradrenaline': 0.4, 'duration': 15},
            
            # GABA系
            'benzodiazepine': {'gaba': 0.7, 'duration': 25},
            
            # グルタミン酸系
            'ketamine': {'glutamate': -0.6, 'duration': 10},
            
            # アセチルコリン系
            'nicotine': {'acetylcholine': 0.5, 'dopamine': 0.3, 'duration': 8},
        }
        
        # 現在活性化している薬物
        self.active_drugs = {}
    
    def apply_drug(self, drug_name: str, dose: float = 1.0):
        """
        薬物を適用
        
        Args:
            drug_name: 薬物名
            dose: 用量（0.0～2.0、1.0が標準）
        """
        if drug_name in self.registered_drugs:
            drug_info = self.registered_drugs[drug_name].copy()
            duration = drug_info.pop('duration')
            
            # 用量に応じて効果と持続時間を調整
            for nt in drug_info:
                drug_info[nt] *= dose
            duration *= max(0.5, min(2.0, dose))
            
            drug_info['duration'] = duration
            drug_info['remaining'] = duration
            
            self.active_drugs[drug_name] = drug_info
            return True
        return False
    
    def register_drug(self, drug_name: str, effects: Dict[str, float], duration: float):
        """
        新しい薬物を登録
        
        Args:
            drug_name: 薬物名
            effects: 神経伝達物質への効果（キー: 神経伝達物質名, 値: 効果の強さ）
            duration: 効果の持続時間
        """
        if drug_name not in self.registered_drugs:
            drug_effects = effects.copy()
            drug_effects['duration'] = duration
            self.registered_drugs[drug_name] = drug_effects
            return True
        return False
    
    def update(self, delta_t: float = 1.0) -> Dict[str, float]:
        """
        薬物効果の更新と現在有効な効果の取得
        
        Args:
            delta_t: 時間ステップ
            
        Returns:
            有効な薬物効果の合計
        """
        # 現在の総合効果
        current_effects = {}
        drugs_to_remove = []
        
        # 各活性薬物の効果を集計
        for drug_name, drug_info in self.active_drugs.items():
            # 残り時間を減少
            drug_info['remaining'] -= delta_t
            
            # 効果の強度を計算（単純な線形減衰モデル）
            intensity = drug_info['remaining'] / drug_info['duration']
            
            # 期限切れの薬物をマーク
            if drug_info['remaining'] <= 0:
                drugs_to_remove.append(drug_name)
                continue
            
            # 各神経伝達物質への効果を加算
            for nt, effect in drug_info.items():
                if nt not in ['duration', 'remaining']:
                    if nt not in current_effects:
                        current_effects[nt] = 0
                    current_effects[nt] += effect * intensity
        
        # 期限切れの薬物を削除
        for drug in drugs_to_remove:
            del self.active_drugs[drug]
        
        return current_effects
    
    def get_active_drugs(self) -> Dict[str, Dict[str, float]]:
        """
        現在活性化している薬物の情報を取得
        
        Returns:
            活性薬物の情報
        """
        return self.active_drugs.copy() 