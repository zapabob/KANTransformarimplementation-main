@echo off
REM BioKANモデルのOptunaによるハイパーパラメータ最適化実行スクリプト

echo BioKANモデルのOptunaによるハイパーパラメータ最適化を開始します
echo ========================================================

REM Pythonバージョンチェック
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>nul
if errorlevel 1 (
    echo エラー: Python 3.11以上が必要です
    exit /b 1
)

REM 出力ディレクトリの作成
mkdir optuna_results 2>nul

REM 各タスクタイプに対してOptunaを実行
echo 分類タスクのハイパーパラメータ最適化中...
python run_biokan_advanced_inference.py --task-type classification --optimize --n-trials 20 --save-dir optuna_results/classification --epochs 10

echo.
echo 回帰タスクのハイパーパラメータ最適化中...
python run_biokan_advanced_inference.py --task-type regression --optimize --n-trials 20 --save-dir optuna_results/regression --epochs 10

echo.
echo 多変量回帰タスクのハイパーパラメータ最適化中...
python run_biokan_advanced_inference.py --task-type multivariate_regression --optimize --n-trials 15 --save-dir optuna_results/multivariate_regression --epochs 8

echo.
echo 時系列予測タスクのハイパーパラメータ最適化中...
python run_biokan_advanced_inference.py --task-type sequence --optimize --n-trials 15 --save-dir optuna_results/sequence --epochs 10

echo.
echo セグメンテーションタスクのハイパーパラメータ最適化中...
python run_biokan_advanced_inference.py --task-type segmentation --optimize --n-trials 10 --save-dir optuna_results/segmentation --epochs 8

echo.
echo 異常検知タスクのハイパーパラメータ最適化中...
python run_biokan_advanced_inference.py --task-type anomaly_detection --optimize --n-trials 15 --save-dir optuna_results/anomaly_detection --epochs 8

echo.
echo ========================================================
echo すべてのハイパーパラメータ最適化が完了しました。
echo 結果は optuna_results ディレクトリに保存されています。
echo. 