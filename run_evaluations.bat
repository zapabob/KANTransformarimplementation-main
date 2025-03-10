@echo off
echo BioKANモデル評価スクリプト
echo ===========================
echo.

REM 出力ディレクトリの作成
mkdir outputs_mnist
mkdir outputs_cifar

REM MNISTデータセットでの評価
echo MNISTデータセットの評価を実行中...
python -m biokan.examples.mnist_evaluation --batch_size 128 --epochs 10 --hidden_dim 128 --num_blocks 2 --attention_type biological --save_dir outputs_mnist --explain
echo.
echo MNIST評価が完了しました。

REM CIFAR-10データセットでの評価
echo CIFAR-10データセットの評価を実行中...
python -m biokan.examples.cifar_evaluation --batch_size 128 --epochs 20 --hidden_dim 256 --num_blocks 3 --attention_type cortical --dropout 0.3 --save_dir outputs_cifar --explain
echo.
echo CIFAR-10評価が完了しました。

echo.
echo 全ての評価が完了しました。
echo 結果は以下のディレクトリに保存されています:
echo  - MNIST: outputs_mnist
echo  - CIFAR: outputs_cifar
echo.

pause 