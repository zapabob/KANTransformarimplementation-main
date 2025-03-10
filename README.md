# BioKAN: 生体模倣コルモゴロフ・アーノルド・ネットワーク

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

BioKANは、コルモゴロフ・アーノルド・ネットワーク（KAN）を拡張して神経科学的原理を統合し、より透明で説明可能な深層学習アーキテクチャを提供するフレームワークです。

## プロジェクト概要

BioKANは以下の特徴を持っています：

1. **神経伝達物質システム**: ドーパミン、セロトニン、ノルアドレナリン、アセチルコリンなどの神経伝達物質の動態をシミュレートし、ネットワークの動作を調整します。

2. **グリア細胞モデル**: アストロサイトとミクログリアなどのグリア細胞が神経回路に与える影響をモデル化し、恒常性維持や情報処理への貢献を表現します。

3. **バイオロジカルアテンション**: 生物学的に着想を得たアテンションメカニズムにより、情報処理の選択性と階層性を実現します。

4. **説明可能AIコンポーネント**: モデルの内部状態を分析し、神経科学的視点から意思決定プロセスを解釈する機能を提供します。

5. **可視化ツール**: 神経活動、神経伝達物質の動態、アテンションパターンなどを可視化するためのツールセットを備えています。

## インストール方法

このリポジトリをクローンし、必要なパッケージをインストールします：

```bash
git clone https://github.com/zapabob/biokanimplimention.git
cd biokan
pip install -r requirements.txt
```

## 使用方法

### 基本的な分類器の作成

BioKANを使用して分類器を作成する基本的な例：

```python
import torch
from biokan.core.biokan_model import create_biokan_classifier

# モデルの作成
model = create_biokan_classifier(
    in_features=784,       # 入力特徴量の数
    hidden_dim=128,        # 隠れ層の次元
    num_classes=10,        # 出力クラス数
    num_blocks=3,          # BioKANブロック数
    attention_type='biological',  # アテンションタイプ
    neuromodulation=True   # 神経調節を有効にする
)

# 入力データの準備
batch_size = 32
x = torch.randn(batch_size, 784)  # 例: MNISTデータ

# 推論
logits = model(x)
predictions = torch.argmax(logits, dim=1)
```

### 説明可能性の利用

モデルの予測を説明する例：

```python
from biokan.xai.explainers import BioKANExplainer
from biokan.visualization.visualizers import visualize_explanation
import matplotlib.pyplot as plt

# 説明者を初期化
explainer = BioKANExplainer(model)

# 入力サンプルを準備
input_sample = torch.randn(1, 784)  # 単一サンプル
target = torch.tensor([3])          # 例: ラベル

# 分析を実行
explanation = explainer.analyze(input_sample, target)

# 反事実的説明を生成
counterfactual = explainer.create_counterfactual(
    input_sample, 
    target_class=5,  # 目標クラス
    max_iterations=50
)

# 説明を可視化
figures = visualize_explanation(explanation)

# 図を表示
for name, fig in figures.items():
    plt.figure(fig.number)
    plt.title(name)
    plt.show()
```

### MNISTデータセットでの例

付属のMNIST分類例を実行：

```bash
python -m biokan.examples.simple_classifier --epochs 5 --hidden_dim 128 --num_blocks 2 --explain
```

オプション:
- `--batch_size`: バッチサイズ（デフォルト: 128）
- `--epochs`: 訓練エポック数（デフォルト: 5）
- `--lr`: 学習率（デフォルト: 0.001）
- `--hidden_dim`: 隠れ層の次元（デフォルト: 128）
- `--num_blocks`: BioKANブロック数（デフォルト: 2）
- `--attention_type`: アテンションタイプ（biological, cortical, hierarchical）
- `--neuromodulation`: 神経調節を有効にする
- `--save_dir`: 出力保存ディレクトリ
- `--explain`: 予測の説明を生成する
- `--num_explain`: 説明するサンプル数（デフォルト: 5）

## モジュール構成

プロジェクトは以下のモジュールで構成されています：

- `biokan.core`: コアモデルと基本レイヤー
  - `biokan_model.py`: BioKANモデルの実装
  - `layers.py`: 基本的なニューラルネットワーク層

- `biokan.neuro`: 神経生物学的コンポーネント
  - `neuromodulators.py`: 神経伝達物質システム
  - `glial_cells.py`: グリア細胞の実装

- `biokan.xai`: 説明可能AIコンポーネント
  - `explainers.py`: モデル説明器

- `biokan.visualization`: 可視化ツール
  - `visualizers.py`: 神経活動と説明可能性の可視化

- `biokan.utils`: ユーティリティ関数

- `biokan.examples`: 使用例
  - `simple_classifier.py`: MNISTデータセットでの分類例

## 神経伝達物質の効果

BioKANでは以下の神経伝達物質の影響を科学文献に基づいて正規化し、モデル化しています：

### 興奮性神経伝達物質

| 神経伝達物質 | 主な効果 | 正規化範囲 | 引用文献 |
|------------|--------|----------|---------|
| グルタミン酸 | 主要な興奮性神経伝達物質。学習、記憶形成、シナプス可塑性に重要。 | 0.0-1.0 | Danbolt, N. C. (2001). Glutamate uptake. Progress in Neurobiology, 65(1), 1-105. |
| アセチルコリン | 自律神経系および中枢神経系での情報伝達。注意、学習、記憶に関与。 | 0.0-0.8 | Picciotto, M. R., et al. (2012). Acetylcholine as a neuromodulator: cholinergic signaling shapes nervous system function and behavior. Neuron, 76(1), 116-129. |

### 抑制性神経伝達物質

| 神経伝達物質 | 主な効果 | 正規化範囲 | 引用文献 |
|------------|--------|----------|---------|
| GABA | 主要な抑制性神経伝達物質。神経活動の調節とバランス維持に必須。 | -1.0-0.0 | Olsen, R. W., & Sieghart, W. (2009). GABA A receptors: subtypes provide diversity of function and pharmacology. Neuropharmacology, 56(1), 141-148. |

### モノアミン神経伝達物質

| 神経伝達物質 | 主な効果 | 正規化範囲 | 引用文献 |
|------------|--------|----------|---------|
| ドーパミン | 報酬、動機付け、運動制御に関与。報酬予測誤差信号を生成。 | -0.2-1.0 | Schultz, W. (2007). Multiple dopamine functions at different time courses. Annual Review of Neuroscience, 30, 259-288. |
| セロトニン | 気分、食欲、睡眠、認知機能の調節に関与。 | -0.5-0.8 | Berger, M., et al. (2009). The expanded biology of serotonin. Annual Review of Medicine, 60, 355-366. |
| ノルアドレナリン | 注意、覚醒、ストレス反応に関与。情報の選択性を高める。 | -0.1-0.9 | Sara, S. J. (2009). The locus coeruleus and noradrenergic modulation of cognition. Nature Reviews Neuroscience, 10(3), 211-223. |

## 向精神薬（中枢神経作用薬）の効果

BioKANでは以下の向精神薬の効果もシミュレートできます：

### 抗精神病薬

| 薬剤タイプ | 主な効果 | 神経伝達物質への影響 | 正規化係数 | 引用文献 |
|-----------|--------|-------------------|-----------|---------|
| 第一世代（定型） | ドーパミンD2受容体の遮断 | DA: -0.7〜-0.9 | 0.8 | Kapur, S., & Mamo, D. (2003). Half a century of antipsychotics and still a central role for dopamine D2 receptors. Progress in Neuro-Psychopharmacology and Biological Psychiatry, 27(7), 1081-1090. |
| 第二世代（非定型） | D2/5-HT2A受容体遮断 | DA: -0.4〜-0.7, 5-HT: -0.3〜-0.6 | 0.6 | Meltzer, H. Y., & Massey, B. W. (2011). The role of serotonin receptors in the action of atypical antipsychotic drugs. Current Opinion in Pharmacology, 11(1), 59-67. |

### 抗うつ薬

| 薬剤タイプ | 主な効果 | 神経伝達物質への影響 | 正規化係数 | 引用文献 |
|-----------|--------|-------------------|-----------|---------|
| SSRI | セロトニン再取り込み阻害 | 5-HT: +0.4〜+0.7 | 0.7 | Vaswani, M., et al. (2003). Role of selective serotonin reuptake inhibitors in psychiatric disorders. Progress in Neuro-Psychopharmacology and Biological Psychiatry, 27(1), 85-102. |
| SNRI | セロトニン・ノルアドレナリン再取り込み阻害 | 5-HT: +0.3〜+0.6, NA: +0.3〜+0.6 | 0.65 | Stahl, S. M., et al. (2005). SNRIs: their pharmacology, clinical efficacy, and tolerability in comparison with other classes of antidepressants. CNS Spectrums, 10(9), 732-747. |

### 抗不安薬

| 薬剤タイプ | 主な効果 | 神経伝達物質への影響 | 正規化係数 | 引用文献 |
|-----------|--------|-------------------|-----------|---------|
| ベンゾジアゼピン | GABA-A受容体の正のアロステリック調節 | GABA: +0.3〜+0.7 | 0.75 | Möhler, H., et al. (2002). A new benzodiazepine pharmacology. Journal of Pharmacology and Experimental Therapeutics, 300(1), 2-8. |

詳細な情報と実装については、`docs/neurotransmitters_psychotropics.md`と`biokan/neuro/neuromodulators.py`を参照してください。

## グリア細胞による皮質層間時間差処理

BioKANモデルでは、神経科学の知見に基づいたグリア細胞（特にアストロサイト）の働きを実装し、皮質層間の時間差処理を行っています。

### 実装された機能

1. **アストロサイトによる時間統合**
   - 皮質の各層（6層構造）における神経活動の時間的差異を検出
   - カルシウムウェーブによる活動の時空間的拡散をシミュレート
   - 層間の情報伝達における時間遅延を神経科学的にモデル化

2. **層間時間差変調メカニズム**
   - 上位層（トップダウン）と下位層（ボトムアップ）の情報処理の方向性を区別
   - 各層の時間遅延特性を生物学的に正確に表現
   - 層間の時間差に基づく変調効果を計算

3. **皮質層間接続の動的調整**
   - 各層間の活動履歴を追跡
   - 時間的な活動変化を検出して層間の調節信号として利用
   - 学習可能な層間接続強度パラメータによる適応的な調整

### 神経科学的背景

グリア細胞（特にアストロサイト）は単なる支持細胞ではなく、神経情報処理に積極的に関与しています：

- **時間的統合能力**: アストロサイトは数秒から数分の時間スケールで活動を統合できる
- **カルシウムシグナリング**: アストロサイト内のCa²⁺波は神経活動の時空間パターンを反映
- **シナプス調節**: 神経伝達物質の取り込みと放出によるシナプス強度の動的制御
- **三者間シナプス**: 神経-グリア-神経の相互作用による情報処理

### 使用方法

```python
from biokan.core.biokan_model import create_biokan_classifier

# 皮質層間時間差処理を備えたBioKANモデルを作成
model = create_biokan_classifier(
    in_features=784,  # 入力特徴量
    hidden_dim=128,   # 隠れ層次元
    num_classes=10,   # 出力クラス数
    num_blocks=3,     # BioKANブロック数
    attention_type='biological',  # 生物学的アテンション使用
    neuromodulation=True  # グリア細胞の時間差処理を有効化
)

# モデル訓練
# ...

# 推論
outputs = model(inputs)
```

### 時間差処理の効果

グリア細胞による皮質層間の時間差処理は以下のような効果をもたらします：

1. **予測的処理**: 過去の層間活動パターンに基づく予測的情報処理
2. **時間的文脈の統合**: 異なる時間スケールの情報を統合
3. **リズミックな活動の調整**: 脳波のような振動パターンのサポート
4. **情報の階層的処理**: 低次から高次への情報の段階的な変換の調整

このモジュールにより、BioKANモデルはより生物学的に忠実な時間的情報処理が可能になり、時系列データや時間的依存性のある問題に対してより効果的に対応できます。

## 引用

このプロジェクトを引用する場合は、以下の形式を使用してください：

```
@software{biokan2023,
  author = {BioKAN Team},
  title = {BioKAN: Biologically-inspired Kolmogorov-Arnold Network},
  year = {2023},
  url = {https://github.com/zapabob/biokan}
}
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細はLICENSEファイルを参照してください。

## 謝辞

このプロジェクトは元のKANアーキテクチャに基づいており、神経科学的知見を組み込んで拡張したものです。研究コミュニティからのフィードバックと貢献に感謝します。

## ハイパーパラメータ最適化

BioKANモデルは、Optunaを使用したハイパーパラメータの自動最適化機能をサポートしています。以下のハイパーパラメータが最適化の対象となります：

- `hidden_dim`: ネットワークの隠れ層の次元数
- `num_blocks`: モデル内のBioKANブロック数
- `attention_type`: アテンションメカニズムのタイプ（biological, cortical, hierarchical）
- `dropout`: ドロップアウト率
- `learning_rate`: 学習率
- `weight_decay`: 重み減衰パラメータ
- `neuromodulation`: 神経調節機能の有効/無効

### 最適化の実行方法

Windowsの場合：
```
run_optuna.bat
```

Linux/Macの場合：
```
chmod +x run_optuna.sh
./run_optuna.sh
```

最適化結果は `optuna_results` ディレクトリに保存され、以下のファイルが含まれます：

- `optuna_results.json`: 最適化の結果とベストパラメータ
- `param_importance.json`: パラメータの重要度
- `param_importance.png`: パラメータ重要度のグラフ
- `optimization_history.png`: 最適化の履歴グラフ
- `parallel_coordinate.png`: 並行座標プロット
- `contour.png`: パラメータ空間の等高線プロット
- `class_accuracy.png`: クラスごとの精度グラフ（評価オプション有効時）

### カスタム最適化の実行

特定のパラメータでカスタム最適化を実行する場合：

```
python biokan/examples/mnist_optuna.py --batch_size 128 --epochs 15 --n_trials 30 --save_dir ./my_optuna_results --eval_best
```

```
python biokan/examples/cifar_optuna.py --batch_size 64 --epochs 20 --n_trials 30 --save_dir ./my_optuna_results --eval_best
```

## 転移学習と高度な推論タスク

BioKANモデルは拡張された転移学習機能により、様々な推論問題に対応できるようになりました。事前学習済みモデルを基盤として、以下のタスクに特化した推論が可能です：

### 対応する推論タスク

1. **分類（Classification）**
   - 標準的な多クラス分類問題
   - Fashion-MNISTなどの画像分類タスク

2. **回帰（Regression）**
   - 単一出力の回帰問題
   - 連続値予測タスク

3. **多変量回帰（Multivariate Regression）**
   - 複数出力の回帰問題
   - 同時に複数の目標変数を予測

4. **時系列予測（Sequence）**
   - 時系列データの予測
   - LSTM層による系列処理

5. **画像セグメンテーション（Segmentation）**
   - ピクセルレベルの分類
   - UNet風のデコーダーによる特徴マップ復元

6. **異常検知（Anomaly Detection）**
   - 自己符号化器アプローチによる異常検出
   - 正常データからの逸脱を検知

### 使用方法

```bash
# 基本的な使用方法
python run_biokan_advanced_inference.py --task-type classification

# 回帰タスクの例
python run_biokan_advanced_inference.py --task-type regression --epochs 15

# セグメンテーションタスクの例
python run_biokan_advanced_inference.py --task-type segmentation --batch-size 32
```

### 転移学習モデルの特徴

- **生物学的神経伝達物質の活用**: 各タスクに応じて神経伝達物質レベルを動的に調整
- **説明可能性**: 予測結果に対する説明機能を提供
- **特徴抽出**: 事前学習済みモデルの知識を活用した効率的な特徴抽出
- **CUDA 12最適化**: 最新のGPUアクセラレーションに対応

## Optunaによるハイパーパラメータ最適化

BioKANモデルはOptunaを使用したハイパーパラメータの自動最適化をサポートしています。これにより、様々な推論タスクに対して最適なモデル構成を見つけることができます。

### 最適化されるハイパーパラメータ

- **学習率（learning_rate）**: 1e-5〜1e-2の範囲で対数スケールで探索
- **ドロップアウト率（dropout）**: 0.1〜0.5の範囲で探索
- **事前学習済み層の凍結（freeze_layers）**: TrueまたはFalse
- **タスク固有パラメータ**: タスクの種類に応じた特殊パラメータ

### 最適化の実行方法

Windows環境:
```
run_optuna_biokan.bat
```

Linux/macOS環境:
```
chmod +x run_optuna_biokan.sh
./run_optuna_biokan.sh
```

単一タスクの最適化:
```
python run_biokan_advanced_inference.py --task-type regression --optimize --n-trials 30
```

### 最適化の結果

最適化の結果は以下のファイルとして保存されます：

- **最適化されたモデル**: `optuna_results/{task_type}/optimized_{task_type}_model.pth`
- **最適パラメータ**: `optuna_results/{task_type}/optimized_{task_type}_params.json`
- **評価指標**: `optuna_results/{task_type}/{task_type}_results.json`
- **最適化履歴グラフ**: `optimization_history.png`
- **パラメータ重要度**: `param_importance.png`

### 重要な依存関係

- **Optuna**: ハイパーパラメータ最適化フレームワーク
- **Plotly**: 最適化結果の可視化
- **scikit-learn**: データセット生成とモデル評価

```bash
pip install optuna plotly scikit-learn
```

## 要件

- Python 3.11+
- PyTorch 2.0+
- Optuna 4.0+
- scikit-learn 1.2+
- matplotlib 3.7+
- seaborn 0.12+
- numpy 1.24+
- tqdm 4.64+

## インストール

```bash
pip install -r requirements.txt
```

## ライセンス

MIT License

## CUDA 12対応と進捗表示機能の追加

### 主な機能強化

1. **CUDA 12対応**
   - CUDA 12環境での最適化設定を追加
   - TensorFloat-32の活用による演算高速化
   - JITコンパイル最適化の適用
   - 各種CUDA関連パラメータの最適化

2. **tqdmによる進捗表示**
   - トレーニングと検証のプログレスバー表示
   - 各バッチの処理状況をリアルタイムで確認可能
   - 損失、精度、神経伝達物質レベルなどの指標をリアルタイム表示

3. **決定係数（R²）の詳細表示**
   - 各エポックでの訓練・検証データに対するR²値の計算と表示
   - R²値の解釈に関する説明の追加
   - 過学習検出のためのR²差分評価
   - 決定係数の可視化グラフの生成

### 使用方法

新しい実行スクリプト `run_training_with_cuda12.py` を使用して訓練を実行できます：

```bash
# 基本的な使用方法
python run_training_with_cuda12.py

# オプション指定
python run_training_with_cuda12.py --epochs 10 --batch-size 64 --lr 0.001
```

### 利用可能なオプション

- `--batch-size`: バッチサイズ（デフォルト: 64）
- `--epochs`: 訓練エポック数（デフォルト: 10）
- `--lr`: 学習率（デフォルト: 0.001）
- `--weight-decay`: 重み減衰（デフォルト: 1e-5）
- `--hidden-dim`: 隠れ層の次元（デフォルト: 128）
- `--num-blocks`: BioKANブロック数（デフォルト: 3）
- `--save-dir`: モデル保存ディレクトリ（デフォルト: biokan_trained_models）
- `--gpu-clear-cache`: 各エポック後にGPUキャッシュをクリア（フラグ）

### 決定係数（R²）について

決定係数はモデルの予測精度を評価する重要な指標で、以下のように解釈できます：

- R² = 1.0: モデルが完全に予測できている（理想的）
- R² > 0.7: 強い説明力を持つモデル
- R² > 0.5: 中程度の説明力を持つモデル
- R² < 0.3: 弱い説明力（改善の余地あり）

### 注意事項

- CUDA 12が利用可能な環境では自動的に最適化が適用されます
- CPU環境でも互換性を保持して動作します
- 最適なパフォーマンスのためにGPU環境の使用を推奨します
