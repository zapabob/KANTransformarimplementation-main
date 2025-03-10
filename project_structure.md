biokan/
│
├── core/                   # コアモデル実装
│   ├── __init__.py
│   ├── layers.py           # 基本レイヤー定義
│   ├── attention.py        # マルチアテンション機構
│   ├── transformer.py      # Transformer互換モデル
│   ├── kan_model.py        # KANモデル基本実装
│   └── biokan_model.py     # BioKAN拡張モデル
│
├── neuro/                  # 神経生物学的コンポーネント
│   ├── __init__.py
│   ├── neuromodulators.py  # 神経伝達物質シミュレーション
│   ├── glia.py             # グリア細胞（アストロサイト・ミクログリア）
│   ├── cortex.py           # 大脳皮質層構造
│   └── pharmacology.py     # 薬理学的効果
│
├── xai/                    # 説明可能AIコンポーネント
│   ├── __init__.py
│   ├── explainer.py        # モデル説明生成器
│   ├── counterfactual.py   # 反実仮想分析
│   ├── attribution.py      # 特徴帰属分析
│   └── importance.py       # 重要度分析
│
├── visualization/          # 可視化コンポーネント
│   ├── __init__.py
│   ├── neurovis.py         # 神経活動可視化
│   ├── attention_vis.py    # アテンション可視化
│   ├── dashboards.py       # インタラクティブダッシュボード
│   └── animation.py        # アニメーション生成
│
├── utils/                  # ユーティリティ関数
│   ├── __init__.py
│   ├── data.py             # データ処理
│   ├── training.py         # 訓練ヘルパー
│   ├── evaluation.py       # 評価ヘルパー
│   └── statistics.py       # 統計分析
│
├── examples/               # 使用例とデモ
│   ├── basic_classification.py    # 基本分類タスク
│   ├── time_series_prediction.py  # 時系列予測
│   ├── explanation_demo.py        # 説明可能性デモ
│   ├── pharmacological_sim.py     # 薬理学的シミュレーション
│   ├── statistical_experiment.py  # 統計的実験
│   └── README.md                  # デモの実行方法
│
├── tests/                  # テストコード
│   ├── test_core.py
│   ├── test_neuro.py
│   ├── test_xai.py
│   └── test_visualization.py
│
├── docs/                   # ドキュメント
│   ├── architecture.md     # アーキテクチャ説明
│   ├── neuro_components.md # 神経コンポーネント説明
│   ├── xai_features.md     # 説明可能性機能
│   └── api_reference.md    # API参照
│
├── setup.py                # パッケージインストールスクリプト
├── requirements.txt        # 依存関係リスト
├── README.md               # プロジェクト概要
└── LICENSE                 # ライセンス情報
``` 