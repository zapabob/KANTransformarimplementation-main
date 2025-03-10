"""
可視化関連のユーティリティ関数
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from typing import Optional

def setup_japanese_fonts(font_path: Optional[str] = None) -> None:
    """日本語フォントを設定する

    Args:
        font_path: フォントファイルのパス（オプション）
    """
    # デフォルトのフォントパスを設定
    if font_path is None:
        # Windowsの場合
        if os.name == 'nt':
            font_path = 'C:/Windows/Fonts/meiryo.ttc'
        # macOSの場合
        elif os.name == 'posix':
            font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
        # その他のOSの場合
        else:
            font_path = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
    
    # フォントファイルが存在する場合のみ設定
    if os.path.exists(font_path):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Meiryo', 'Hiragino Sans', 'Yu Gothic', 'MS Gothic']
        
        # フォントを登録
        fm.fontManager.addfont(font_path)
        
        print(f"日本語フォントを設定しました: {font_path}")
    else:
        print(f"警告: フォントファイルが見つかりません: {font_path}")
        print("デフォルトのフォントを使用します。")

def set_plot_style(
    figsize: tuple = (10, 6),
    dpi: int = 100,
    grid: bool = True,
    style: str = 'seaborn'
) -> None:
    """プロットのスタイルを設定する

    Args:
        figsize: 図のサイズ（幅, 高さ）
        dpi: 解像度
        grid: グリッドの表示有無
        style: matplotlibのスタイル
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    
    if grid:
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.color'] = '#cccccc'
        plt.rcParams['grid.linestyle'] = '--'

def create_training_plot(
    train_losses: list,
    val_losses: list,
    train_accuracies: list,
    val_accuracies: list,
    title: str = '学習の進捗'
) -> None:
    """学習の進捗をプロットする

    Args:
        train_losses: 訓練損失の履歴
        val_losses: 検証損失の履歴
        train_accuracies: 訓練精度の履歴
        val_accuracies: 検証精度の履歴
        title: グラフのタイトル
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # 損失のプロット
    ax1.plot(epochs, train_losses, 'b-', label='訓練損失')
    ax1.plot(epochs, val_losses, 'r-', label='検証損失')
    ax1.set_title('損失の推移')
    ax1.set_xlabel('エポック')
    ax1.set_ylabel('損失')
    ax1.legend()
    ax1.grid(True)
    
    # 精度のプロット
    ax2.plot(epochs, train_accuracies, 'b-', label='訓練精度')
    ax2.plot(epochs, val_accuracies, 'r-', label='検証精度')
    ax2.set_title('精度の推移')
    ax2.set_xlabel('エポック')
    ax2.set_ylabel('精度 (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show() 