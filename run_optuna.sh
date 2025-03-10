#!/bin/bash
# BioKANモデルのハイパーパラメータ最適化を実行するシェルスクリプト

echo "BioKANモデルのハイパーパラメータ最適化を開始します..."

# 出力ディレクトリの作成
mkdir -p optuna_results/mnist
mkdir -p optuna_results/cifar

# MNISTデータセットでの最適化
echo "MNISTデータセットでの最適化を開始します..."
python biokan/examples/mnist_optuna.py --batch_size 256 --epochs 10 --n_trials 20 --save_dir ./optuna_results/mnist --eval_best
if [ $? -ne 0 ]; then
    echo "MNISTデータセットでの最適化に失敗しました。"
    exit 1
fi

# CIFAR-10データセットでの最適化
echo "CIFAR-10データセットでの最適化を開始します..."
python biokan/examples/cifar_optuna.py --batch_size 128 --epochs 15 --n_trials 20 --save_dir ./optuna_results/cifar --eval_best
if [ $? -ne 0 ]; then
    echo "CIFAR-10データセットでの最適化に失敗しました。"
    exit 1
fi

echo "全ての最適化プロセスが完了しました。"
echo "結果は以下のディレクトリに保存されています："
echo "- MNISTの結果: ./optuna_results/mnist"
echo "- CIFAR-10の結果: ./optuna_results/cifar" 