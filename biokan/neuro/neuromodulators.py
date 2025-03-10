"""
神経伝達物質システム - BioKANモデルの神経調節機構

このモジュールは、様々な神経伝達物質の動態と効果をシミュレートします。
科学文献に基づいて正規化された効果を実装しています。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque


class Neurotransmitter:
    """神経伝達物質の基本クラス"""
    
    def __init__(
        self, 
        name: str,
        initial_level: float = 0.0,
        min_level: float = -1.0,
        max_level: float = 1.0,
        decay_rate: float = 0.1,
        references: List[str] = None
    ):
        """
        Args:
            name: 神経伝達物質の名前
            initial_level: 初期レベル
            min_level: 最小レベル
            max_level: 最大レベル
            decay_rate: 自然減衰率
            references: 関連する文献リスト
        """
        self.name = name
        self.level = initial_level
        self.min_level = min_level
        self.max_level = max_level
        self.decay_rate = decay_rate
        self.baseline = initial_level
        self.references = references or []
        
        # 効果の強さを調整する係数
        self.effectiveness = 1.0
    
    def update(self, delta: float = 0.0) -> float:
        """
        神経伝達物質レベルを更新
        
        Args:
            delta: レベル変化量
        
        Returns:
            更新後のレベル
        """
        # ベースラインに向かって減衰
        decay = self.decay_rate * (self.level - self.baseline)
        
        # レベル更新
        self.level = self.level + delta - decay
        
        # 範囲内に収める
        self.level = max(self.min_level, min(self.max_level, self.level))
        
        return self.level
    
    def get_normalized_level(self) -> float:
        """
        正規化されたレベルを取得（0〜1の範囲）
        
        Returns:
            正規化されたレベル
        """
        return (self.level - self.min_level) / (self.max_level - self.min_level)
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.level:.2f} [{self.min_level:.1f}, {self.max_level:.1f}]"


class Dopamine(Neurotransmitter):
    """ドーパミン神経伝達物質
    
    報酬、動機付け、運動制御に関与。
    報酬予測誤差信号を生成。
    
    References:
        Schultz, W. (2007). Multiple dopamine functions at different time courses. 
        Annual Review of Neuroscience, 30, 259-288.
    """
    
    def __init__(self, initial_level: float = 0.0):
        super().__init__(
            name="Dopamine",
            initial_level=initial_level,
            min_level=-0.2,  # 報酬予測誤差が負の場合
            max_level=1.0,   # 予期せぬ報酬時の最大活性
            decay_rate=0.2,   # ドーパミンは比較的早く減衰
            references=["Schultz, 2007"]
        )
    
    def modulate_learning_rate(self, base_lr: float) -> float:
        """学習率を調整
        
        Args:
            base_lr: 基本学習率
        
        Returns:
            調整された学習率
        """
        # ドーパミンレベルに基づく学習率調整（正のレベルで増加）
        return base_lr * (1.0 + 0.5 * max(0, self.level * self.effectiveness))
    
    def compute_reward_prediction_error(self, expected: float, actual: float) -> float:
        """報酬予測誤差を計算
        
        Args:
            expected: 期待報酬
            actual: 実際の報酬
        
        Returns:
            報酬予測誤差
        """
        error = actual - expected
        # 報酬予測誤差に基づいてドーパミンレベルを更新
        self.update(error * 0.3)
        return error


class Serotonin(Neurotransmitter):
    """セロトニン神経伝達物質
    
    気分、食欲、睡眠、認知機能の調節に関与。
    長期的な価値と報酬の処理に影響。
    
    References:
        Berger, M., Gray, J. A., & Roth, B. L. (2009). The expanded biology of serotonin. 
        Annual Review of Medicine, 60, 355-366.
    """
    
    def __init__(self, initial_level: float = 0.0):
        super().__init__(
            name="Serotonin",
            initial_level=initial_level,
            min_level=-0.5,  # 低セロトニン状態
            max_level=0.8,   # 高セロトニン状態
            decay_rate=0.05,  # セロトニンはゆっくり減衰
            references=["Berger et al., 2009"]
        )
    
    def modulate_long_term_value(self, base_value: float) -> float:
        """長期的な価値評価を調整
        
        Args:
            base_value: 基本評価値
        
        Returns:
            調整された評価値
        """
        # セロトニンレベルに基づく長期的価値の調整
        return base_value * (1.0 + 0.3 * self.level * self.effectiveness)
    
    def regulate_mood(self, base_mood: float) -> float:
        """気分状態を調整
        
        Args:
            base_mood: 基本気分状態
        
        Returns:
            調整された気分状態
        """
        # セロトニンレベルは気分に直接影響
        return base_mood + (0.4 * self.level * self.effectiveness)


class Noradrenaline(Neurotransmitter):
    """ノルアドレナリン（ノルエピネフリン）神経伝達物質
    
    注意、覚醒、ストレス反応に関与。
    情報の選択性を高める。
    
    References:
        Sara, S. J. (2009). The locus coeruleus and noradrenergic modulation of cognition. 
        Nature Reviews Neuroscience, 10(3), 211-223.
    """
    
    def __init__(self, initial_level: float = 0.0):
        super().__init__(
            name="Noradrenaline",
            initial_level=initial_level,
            min_level=-0.1,  # 低覚醒状態
            max_level=0.9,   # 高覚醒状態（ストレス時）
            decay_rate=0.15,  # 中程度の減衰率
            references=["Sara, 2009"]
        )
    
    def modulate_attention(self, base_attention: float) -> float:
        """注意機構を調整
        
        Args:
            base_attention: 基本注意レベル
        
        Returns:
            調整された注意レベル
        """
        # ノルアドレナリンレベルに基づく注意の調整
        return base_attention * (1.0 + 0.4 * self.level * self.effectiveness)
    
    def reduce_noise(self, signal: torch.Tensor, noise_level: float) -> torch.Tensor:
        """信号のノイズを低減
        
        Args:
            signal: 入力信号
            noise_level: 基本ノイズレベル
        
        Returns:
            ノイズ低減された信号
        """
        # ノルアドレナリンレベルが高いほどノイズが減少
        effective_noise = noise_level * (1.0 - 0.5 * max(0, self.level * self.effectiveness))
        
        if effective_noise > 0:
            noise = torch.randn_like(signal) * effective_noise
            return signal + noise
        return signal


class Acetylcholine(Neurotransmitter):
    """アセチルコリン神経伝達物質
    
    自律神経系および中枢神経系での情報伝達。
    注意、学習、記憶に関与。
    
    References:
        Picciotto, M. R., Higley, M. J., & Mineur, Y. S. (2012). Acetylcholine as a 
        neuromodulator: cholinergic signaling shapes nervous system function and behavior. 
        Neuron, 76(1), 116-129.
    """
    
    def __init__(self, initial_level: float = 0.0):
        super().__init__(
            name="Acetylcholine",
            initial_level=initial_level,
            min_level=0.0,   # 最低レベル
            max_level=0.8,   # 最大レベル
            decay_rate=0.1,   # 中程度の減衰率
            references=["Picciotto et al., 2012"]
        )
    
    def enhance_encoding(self, base_encoding: float) -> float:
        """記憶のエンコーディングを強化
        
        Args:
            base_encoding: 基本エンコーディング強度
        
        Returns:
            強化されたエンコーディング強度
        """
        # アセチルコリンレベルに基づくエンコーディングの強化
        return base_encoding * (1.0 + 0.6 * self.level * self.effectiveness)
    
    def modulate_synaptic_plasticity(self, base_plasticity: float) -> float:
        """シナプス可塑性を調整
        
        Args:
            base_plasticity: 基本可塑性
        
        Returns:
            調整された可塑性
        """
        # アセチルコリンはシナプス可塑性を促進
        return base_plasticity * (1.0 + 0.4 * self.level * self.effectiveness)


class GABA(Neurotransmitter):
    """GABA（γ-アミノ酪酸）神経伝達物質
    
    主要な抑制性神経伝達物質。
    神経活動の調節とバランス維持に必須。
    
    References:
        Olsen, R. W., & Sieghart, W. (2009). GABA A receptors: subtypes provide 
        diversity of function and pharmacology. Neuropharmacology, 56(1), 141-148.
    """
    
    def __init__(self, initial_level: float = 0.0):
        super().__init__(
            name="GABA",
            initial_level=initial_level,
            min_level=-1.0,  # 強い抑制
            max_level=0.0,   # 抑制なし
            decay_rate=0.1,   # 中程度の減衰率
            references=["Olsen & Sieghart, 2009"]
        )
    
    def inhibit_activity(self, activity: torch.Tensor) -> torch.Tensor:
        """神経活動を抑制
        
        Args:
            activity: 神経活動
        
        Returns:
            抑制された活動
        """
        # GABAレベルに基づく活動の抑制（負のレベルで抑制が強まる）
        inhibition_factor = 1.0 + (self.level * self.effectiveness)
        return activity * inhibition_factor


class Glutamate(Neurotransmitter):
    """グルタミン酸神経伝達物質
    
    主要な興奮性神経伝達物質。
    学習、記憶形成、シナプス可塑性に重要。
    
    References:
        Danbolt, N. C. (2001). Glutamate uptake. Progress in Neurobiology, 65(1), 1-105.
    """
    
    def __init__(self, initial_level: float = 0.0):
        super().__init__(
            name="Glutamate",
            initial_level=initial_level,
            min_level=0.0,   # 最低レベル
            max_level=1.0,   # 最大レベル
            decay_rate=0.15,  # 中程度の減衰率
            references=["Danbolt, 2001"]
        )
    
    def enhance_activity(self, activity: torch.Tensor) -> torch.Tensor:
        """神経活動を増強
        
        Args:
            activity: 神経活動
        
        Returns:
            増強された活動
        """
        # グルタミン酸レベルに基づく活動の増強
        excitation_factor = 1.0 + (0.5 * self.level * self.effectiveness)
        return activity * excitation_factor
    
    def facilitate_long_term_potentiation(self, base_ltp: float) -> float:
        """長期増強（LTP）を促進
        
        Args:
            base_ltp: 基本LTP強度
        
        Returns:
            促進されたLTP強度
        """
        # グルタミン酸はLTPを促進
        return base_ltp * (1.0 + 0.7 * self.level * self.effectiveness)


class NeuromodulatorSystem(nn.Module):
    """神経伝達物質システム
    
    複数の神経伝達物質を組み合わせて管理し、
    ネットワークの動作を調整するシステム。
    """
    
    def __init__(self):
        super().__init__()
        self.register_buffer('dopamine', torch.zeros(1))
        self.register_buffer('serotonin', torch.zeros(1))
        self.register_buffer('noradrenaline', torch.zeros(1))
        self.register_buffer('acetylcholine', torch.zeros(1))
        
    def update(self, stimuli: Optional[Dict[str, torch.Tensor]] = None, delta_t: float = 1.0):
        if stimuli is None:
            stimuli = {}
            
        # 各神経伝達物質の更新
        self.dopamine = self._update_level(self.dopamine, stimuli.get('dopamine', 0.0), delta_t)
        self.serotonin = self._update_level(self.serotonin, stimuli.get('serotonin', 0.0), delta_t)
        self.noradrenaline = self._update_level(self.noradrenaline, stimuli.get('noradrenaline', 0.0), delta_t)
        self.acetylcholine = self._update_level(self.acetylcholine, stimuli.get('acetylcholine', 0.0), delta_t)
        
    def _update_level(self, current: torch.Tensor, stimulus: float, delta_t: float) -> torch.Tensor:
        decay = 0.1 * delta_t
        return current * (1 - decay) + torch.tensor(stimulus).to(current.device) * delta_t
        
    def get_state(self) -> Dict[str, torch.Tensor]:
        return {
            'dopamine': self.dopamine,
            'serotonin': self.serotonin,
            'noradrenaline': self.noradrenaline,
            'acetylcholine': self.acetylcholine
        }
    
    def apply_drug_effect(self, drug_type: str, dose: float = 1.0) -> None:
        """
        薬理学的効果を適用
        
        Args:
            drug_type: 薬剤タイプ
            dose: 用量（0.0〜1.0）
        """
        # 用量を有効範囲に制限
        dose = max(0.0, min(1.0, dose))
        
        if drug_type == "SSRI":
            # 選択的セロトニン再取り込み阻害薬
            self.serotonin = self._update_level(self.serotonin, 0.5 * dose, 1.0)
            
        elif drug_type == "SNRI":
            # セロトニン・ノルアドレナリン再取り込み阻害薬
            self.serotonin = self._update_level(self.serotonin, 0.4 * dose, 1.0)
            self.noradrenaline = self._update_level(self.noradrenaline, 0.4 * dose, 1.0)
            
        elif drug_type == "TCA":
            # 三環系抗うつ薬
            self.serotonin = self._update_level(self.serotonin, 0.3 * dose, 1.0)
            self.noradrenaline = self._update_level(self.noradrenaline, 0.3 * dose, 1.0)
            self.acetylcholine = self._update_level(self.acetylcholine, -0.3 * dose, 1.0)  # 抗コリン作用
            
        elif drug_type == "typical_antipsychotic":
            # 定型抗精神病薬
            self.dopamine = self._update_level(self.dopamine, -0.7 * dose, 1.0)
            
        elif drug_type == "atypical_antipsychotic":
            # 非定型抗精神病薬
            self.dopamine = self._update_level(self.dopamine, -0.5 * dose, 1.0)
            self.serotonin = self._update_level(self.serotonin, -0.4 * dose, 1.0)
            
        elif drug_type == "benzodiazepine":
            # ベンゾジアゼピン系抗不安薬
            # GABAの効果を増強（レベル自体ではなく有効性を高める）
            self.GABA.effectiveness += 0.6 * dose
            
        elif drug_type == "amphetamine":
            # アンフェタミン系刺激薬
            self.dopamine = self._update_level(self.dopamine, 0.7 * dose, 1.0)
            self.noradrenaline = self._update_level(self.noradrenaline, 0.5 * dose, 1.0)
            
        elif drug_type == "methylphenidate":
            # メチルフェニデート（リタリンなど）
            self.dopamine = self._update_level(self.dopamine, 0.5 * dose, 1.0)
            self.noradrenaline = self._update_level(self.noradrenaline, 0.4 * dose, 1.0)
            
        elif drug_type == "caffeine":
            # カフェイン
            self.noradrenaline = self._update_level(self.noradrenaline, 0.3 * dose, 1.0)
            self.acetylcholine = self._update_level(self.acetylcholine, 0.2 * dose, 1.0)
            
        else:
            raise ValueError(f"Unknown drug type: {drug_type}")
    
    def reset(self) -> None:
        """全ての神経伝達物質を初期状態にリセット"""
        self.dopamine = torch.zeros(1)
        self.serotonin = torch.zeros(1)
        self.noradrenaline = torch.zeros(1)
        self.acetylcholine = torch.zeros(1)
        self.GABA.effectiveness = 1.0
    
    def __repr__(self) -> str:
        return f"NeuromodulatorSystem: dopamine={self.dopamine[0]:.2f}, serotonin={self.serotonin[0]:.2f}, noradrenaline={self.noradrenaline[0]:.2f}, acetylcholine={self.acetylcholine[0]:.2f}"


class PharmacologicalModulator:
    """薬理学的モジュレータ
    
    神経伝達物質システムに対する薬理学的効果をシミュレート
    
    References:
        Bezchlibnyk-Butler, K. Z., & Jeffries, J. J. (2007). Clinical handbook 
        of psychotropic drugs (17th ed.). Hogrefe & Huber Publishers.
    """
    
    def __init__(self, neuromodulator_system: NeuromodulatorSystem):
        """
        Args:
            neuromodulator_system: 対象の神経伝達物質システム
        """
        self.neuromodulator_system = neuromodulator_system
        self.active_drugs = {}  # 現在作用中の薬剤 {名前: (用量, 残り時間)}
        
        # 薬剤情報辞書
        self.drug_info = {
            # 抗うつ薬
            "SSRI": {
                "half_life": 24.0,  # 時間
                "target": "serotonin",
                "effect_size": 0.7,
                "description": "選択的セロトニン再取り込み阻害薬",
                "reference": "Vaswani et al., 2003"
            },
            "SNRI": {
                "half_life": 12.0,
                "target": ["serotonin", "noradrenaline"],
                "effect_size": 0.65,
                "description": "セロトニン・ノルアドレナリン再取り込み阻害薬",
                "reference": "Stahl et al., 2005"
            },
            "TCA": {
                "half_life": 36.0,
                "target": ["serotonin", "noradrenaline", "acetylcholine"],
                "effect_size": 0.5,
                "description": "三環系抗うつ薬",
                "reference": "Gillman, 2007"
            },
            
            # 抗精神病薬
            "typical_antipsychotic": {
                "half_life": 24.0,
                "target": "dopamine",
                "effect_size": 0.8,
                "description": "定型抗精神病薬",
                "reference": "Kapur & Mamo, 2003"
            },
            "atypical_antipsychotic": {
                "half_life": 24.0,
                "target": ["dopamine", "serotonin"],
                "effect_size": 0.6,
                "description": "非定型抗精神病薬",
                "reference": "Meltzer & Massey, 2011"
            },
            
            # 抗不安薬
            "benzodiazepine": {
                "half_life": 12.0,
                "target": "GABA",
                "effect_size": 0.75,
                "description": "ベンゾジアゼピン系抗不安薬",
                "reference": "Möhler et al., 2002"
            },
            
            # 精神刺激薬
            "amphetamine": {
                "half_life": 10.0,
                "target": ["dopamine", "noradrenaline"],
                "effect_size": 0.85,
                "description": "アンフェタミン系刺激薬",
                "reference": "Fleckenstein et al., 2007"
            },
            "methylphenidate": {
                "half_life": 4.0,
                "target": ["dopamine", "noradrenaline"],
                "effect_size": 0.7,
                "description": "メチルフェニデート",
                "reference": "Volkow et al., 2005"
            },
            "caffeine": {
                "half_life": 5.0,
                "target": ["noradrenaline", "acetylcholine"],
                "effect_size": 0.4,
                "description": "カフェイン",
                "reference": "Fredholm et al., 1999"
            }
        }
    
    def apply_drug(self, drug_name: str, dose: float = 1.0, duration: float = None) -> None:
        """
        薬剤を適用
        
        Args:
            drug_name: 薬剤名
            dose: 用量（0.0〜1.0）
            duration: 作用時間（時間）、省略時は半減期の5倍
        """
        if drug_name not in self.drug_info:
            raise ValueError(f"未知の薬剤: {drug_name}")
        
        # 用量を有効範囲に制限
        dose = max(0.0, min(1.0, dose))
        
        # 半減期から作用時間を計算（5倍で99%減衰）
        if duration is None:
            duration = self.drug_info[drug_name]["half_life"] * 5.0
        
        # 直ちに効果を適用
        self.neuromodulator_system.apply_drug_effect(drug_name, dose)
        
        # 進行中の薬剤リストに追加
        self.active_drugs[drug_name] = (dose, duration)
    
    def update(self, time_delta: float = 1.0) -> None:
        """
        時間経過に伴う薬理効果の更新
        
        Args:
            time_delta: 経過時間（時間）
        """
        drugs_to_remove = []
        
        for drug_name, (dose, remaining_time) in self.active_drugs.items():
            # 残り時間を減らす
            new_remaining = remaining_time - time_delta
            
            if new_remaining <= 0:
                # 薬剤の効果が切れた
                drugs_to_remove.append(drug_name)
            else:
                # 新しい用量を計算（半減期に基づく指数関数的減衰）
                half_life = self.drug_info[drug_name]["half_life"]
                decay_factor = 2 ** (-time_delta / half_life)
                new_dose = dose * decay_factor
                
                # 効果を更新
                self.neuromodulator_system.apply_drug_effect(drug_name, new_dose)
                
                # 薬剤情報を更新
                self.active_drugs[drug_name] = (new_dose, new_remaining)
        
        # 期限切れの薬剤を削除
        for drug_name in drugs_to_remove:
            del self.active_drugs[drug_name]
    
    def get_active_drugs(self) -> Dict[str, Tuple[float, float]]:
        """
        現在活性な薬剤のリストを取得
        
        Returns:
            活性な薬剤の辞書 {名前: (用量, 残り時間)}
        """
        return self.active_drugs.copy()
    
    def clear_all_drugs(self) -> None:
        """全ての薬剤効果をクリア"""
        self.active_drugs.clear()
        self.neuromodulator_system.reset()
    
    def get_drug_info(self, drug_name: str = None) -> Dict:
        """
        薬剤情報を取得
        
        Args:
            drug_name: 薬剤名（省略時は全薬剤）
        
        Returns:
            薬剤情報
        """
        if drug_name:
            if drug_name not in self.drug_info:
                raise ValueError(f"未知の薬剤: {drug_name}")
            return self.drug_info[drug_name]
        else:
            return self.drug_info 