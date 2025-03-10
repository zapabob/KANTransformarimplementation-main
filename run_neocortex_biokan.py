"""
NeoCortexBioKAN モデル実行スクリプト

人間の脳を忠実に模倣するモデルを実行し、その性能を評価します。
引用文献に基づいて、最先端の神経科学の知見を取り入れたディープラーニングモデルです。
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# プロジェクトルートへのパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from examples.brain_simulation import train_neocortex_biokan, analyze_brain_simulation
from biokan.core.biokan_model import NeoCortexBioKAN

if __name__ == "__main__":
    print("====================================================================")
    print("  NeoCortexBioKAN - 人間の脳を模倣するニューラルネットワークモデル")
    print("====================================================================")
    print("\n神経科学の最新知見に基づく人工知能モデルを実行します。")
    print("このモデルは以下の脳の特性を模倣しています：")
    print("  * 皮質層構造（6層構造）")
    print("  * 脳波リズム（θ, α, β, γ, δ波）")
    print("  * ワーキングメモリと前頭前皮質機能")
    print("  * 予測符号化（自由エネルギー原理）")
    print("  * デフォルトモードネットワーク")
    print("  * 高次認知機能（抽象思考、アナロジー推論、創造性）")
    print("\n参考文献：")
    print("  - Friston, K. (2010). The free-energy principle: a unified brain theory?")
    print("  - Sporns, O. (2011). Networks of the Brain.")
    print("  - Buzsáki, G. (2006). Rhythms of the Brain.")
    print("  - Goldman-Rakic, P.S. (1995). Cellular basis of working memory.")
    
    try:
        print("\nモデルのトレーニングと評価を開始します...")
        model = train_neocortex_biokan()
        
        # GPUが利用可能なら使用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # データセットの再作成とデータローダーの初期化は関数内で行われます
        from examples.brain_simulation import generate_complex_sequence_data
        from torch.utils.data import DataLoader, TensorDataset
        
        X_test, y_test = generate_complex_sequence_data(n_samples=200, seq_length=64, n_features=16)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print("\n脳シミュレーションの詳細分析を開始します...")
        analyze_brain_simulation(model, test_loader, device)
        
        print("\n実行が完了しました。以下のファイルが生成されました：")
        print("  * neocortex_biokan_model.pth - トレーニング済みモデル")
        print("  * neocortex_biokan_training_history.png - トレーニング履歴")
        print("  * neocortex_biokan_brain_waves.png - 脳波シミュレーション")
        print("  * neocortex_biokan_cortical_layers.png - 皮質層活動")
        print("  * neocortex_biokan_working_memory.png - ワーキングメモリ状態")
        print("  * neocortex_biokan_higher_cognition.png - 高次認知機能分析")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc() 