"""
Fashion MNISTデータセットを用いたBioKANモデルの転移学習
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import time

from cuda_info_manager import print_cuda_info, get_device, setup_japanese_fonts
from biokan_transfer_learning import TransferBioKANModel
from biokan_training import EnhancedBioKANModel

# 日本語フォントの設定
setup_japanese_fonts(verbose=False)

# デバイスの設定
device = get_device()
print_cuda_info(verbose=True)

def train_fashion_mnist():
    """Fashion MNISTデータセットを用いた転移学習を実行"""
    
    # データセットの準備
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # ランダムな水平反転
        transforms.RandomRotation(10),      # ±10度のランダムな回転
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # ランダムな平行移動
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Fashion MNISTデータセットのロード
    train_dataset = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    # データローダーの設定
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,  # バッチサイズを増加
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,  # バッチサイズを増加
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # クラス名の定義
    class_names = ['Tシャツ/トップ', 'ズボン', 'プルオーバー', 'ドレス', 'コート',
                  'サンダル', 'シャツ', 'スニーカー', 'バッグ', 'アンクルブーツ']
    
    # 事前学習済みモデルのパス
    pretrained_path = 'optimized_mnist_classification_model.pth'
    
    # 事前学習済みモデルの読み込み（weights_only=Trueを指定）
    state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
    
    # state_dictのキーから'pretrained_model.'を削除し、必要なキーのみを抽出
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('pretrained_model.'):
            new_key = key.replace('pretrained_model.', '')
            if not key.startswith('pretrained_model.output_layer.'):  # 出力層は除外
                new_state_dict[new_key] = value
    
    print("読み込んだモデルの層:", new_state_dict.keys())
    
    # ベースモデルの作成と重みの読み込み
    base_model = EnhancedBioKANModel()
    try:
        base_model.load_state_dict(new_state_dict, strict=False)
        print("ベースモデルの重みを正常に読み込みました。")
    except Exception as e:
        print(f"警告: モデルの読み込み中にエラーが発生しました: {e}")
        print("初期化済みのモデルを使用して続行します。")
    
    # 転移学習モデルの作成（アーキテクチャの改善）
    model = TransferBioKANModel(
        pretrained_model=base_model,
        task_type='classification',
        num_classes=10,
        freeze_layers=True,
        additional_params={
            'dropout_rate': 0.3,
            'hidden_dim': 512,
            'num_layers': 3
        }
    )
    model = model.to(device)
    
    # 損失関数とオプティマイザーの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
    
    # 改善された学習率スケジューリング
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=30,  # エポック数を増加
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # ウォームアップ期間
        div_factor=25.0,  # 初期学習率の除数
        final_div_factor=1000.0  # 最終学習率の除数
    )
    
    # Early Stoppingの設定
    early_stopping_patience = 5
    early_stopping_counter = 0
    best_accuracy = 0.0
    
    # 訓練ループ
    num_epochs = 30  # エポック数を増加
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'lr': []  # 学習率の履歴を追加
    }
    
    print("\n=== 転移学習開始 ===")
    print(f"データセット: Fashion MNIST")
    print(f"訓練データ数: {len(train_dataset)}")
    print(f"テストデータ数: {len(test_dataset)}")
    print(f"バッチサイズ: {train_loader.batch_size}")
    print(f"エポック数: {num_epochs}")
    print(f"学習率: {optimizer.param_groups[0]['lr']}")
    print(f"デバイス: {device}")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nエポック {epoch+1}/{num_epochs}")
        
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'エポック {epoch+1}/{num_epochs} [訓練]')
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 入力データの形状を変換（[B, C, H, W] -> [B, -1]）
            inputs = inputs.view(inputs.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # テストフェーズ
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'エポック {epoch+1}/{num_epochs} [テスト]')
            for inputs, targets in test_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 入力データの形状を変換（[B, C, H, W] -> [B, -1]）
                inputs = inputs.view(inputs.size(0), -1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                test_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{test_correct/test_total:.4f}'
                })
        
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_correct / test_total
        
        # 学習率の調整
        scheduler.step()
        
        # 履歴の更新
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start_time
        print(f"\nエポック {epoch+1} の結果:")
        print(f"訓練損失: {train_loss:.4f}, 訓練精度: {train_acc:.2%}")
        print(f"テスト損失: {test_loss:.4f}, テスト精度: {test_acc:.2%}")
        print(f"エポック時間: {epoch_time:.1f}秒")
        
        # 最良モデルの保存
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'transfer_model_fashion_mnist_classification.pth')
            print(f'最良モデルを保存しました（精度: {test_acc:.2%}）')
        
        # Early Stoppingのチェック
        if len(history['test_loss']) > 1:  # 2エポック目以降でチェック
            if test_loss > history['test_loss'][-2]:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("\nEarly Stoppingにより学習を停止します。")
                    break
            else:
                early_stopping_counter = 0  # 改善があった場合はカウンタをリセット
    
    # 学習曲線の描画
    plt.figure(figsize=(15, 5))
    
    # 損失の推移
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='訓練')
    plt.plot(history['test_loss'], label='テスト')
    plt.title('損失の推移')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.legend()
    plt.grid(True)
    
    # 精度の推移
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='訓練')
    plt.plot(history['test_acc'], label='テスト')
    plt.title('精度の推移')
    plt.xlabel('エポック')
    plt.ylabel('精度')
    plt.legend()
    plt.grid(True)
    
    # 学習率の推移
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'])
    plt.title('学習率の推移')
    plt.xlabel('エポック')
    plt.ylabel('学習率')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_fashion_mnist.png')
    plt.close()
    
    # クラスごとの性能評価
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    confusion_mat = torch.zeros(10, 10)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # クラスごとの正解数をカウント
            for target, pred in zip(targets, predicted):
                class_correct[target] += (target == pred).item()
                class_total[target] += 1
                confusion_mat[target][pred] += 1
    
    # 結果の保存
    results = {
        'best_accuracy': best_accuracy,
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_test_loss': history['test_loss'][-1],
        'final_test_acc': history['test_acc'][-1],
        'class_names': class_names,
        'class_accuracies': {
            class_names[i]: class_correct[i] / class_total[i]
            for i in range(10)
        },
        'confusion_matrix': confusion_mat.tolist(),
        'training_history': {
            'train_loss': history['train_loss'],
            'test_loss': history['test_loss'],
            'train_acc': history['train_acc'],
            'test_acc': history['test_acc'],
            'learning_rates': history['lr']
        }
    }
    
    with open('fashion_mnist_transfer_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print('\n=== 転移学習が完了しました ===')
    print(f'最終テスト精度: {results["final_test_acc"]:.2%}')
    print(f'最良テスト精度: {best_accuracy:.2%}')
    print('\nクラスごとの精度:')
    for class_name, accuracy in results['class_accuracies'].items():
        print(f'{class_name}: {accuracy:.2%}')

if __name__ == "__main__":
    train_fashion_mnist() 