"""
BioKANモデルを使用した転移学習スクリプト
MNISTで学習したモデルをFashion-MNISTに転移する例
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from contextlib import nullcontext
from sklearn.metrics import accuracy_score, confusion_matrix

# BioKANモデルのインポート
try:
    from biokan.core.biokan_model import create_biokan_classifier, NeuropharmacologicalBioKAN
    from biokan_training import EnhancedBioKANModel
    from biokan_transfer_learning import TransferBioKANModel
    BIOKAN_AVAILABLE = True
except ImportError:
    print("BioKANモジュールをインポートできませんでした。シンプルなMLPモデルを使用します。")
    BIOKAN_AVAILABLE = False

# CUDA関連の設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 最初のGPUを使用
torch.backends.cudnn.benchmark = True     # CUDNNの自動チューナーを有効化
torch.backends.cudnn.deterministic = True # 再現性のため

# 乱数シードを固定
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# GPU利用可能性の確認
if not torch.cuda.is_available():
    print("警告: GPUが利用できません。CPUを使用します。")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"利用可能なGPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Fashion-MNISTデータセットのラベル
fashion_labels = {
    0: 'Tシャツ/トップ',
    1: 'ズボン',
    2: 'プルオーバー',
    3: 'ドレス',
    4: 'コート',
    5: 'サンダル',
    6: 'シャツ',
    7: 'スニーカー',
    8: 'バッグ',
    9: 'アンクルブーツ'
}

# シンプルなMLPクラス（フォールバックとして使用）
class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 転移学習用のTransferMLPクラス
class TransferMLP(nn.Module):
    def __init__(self, pretrained_model, num_classes=10, freeze_base=True):
        super().__init__()
        # 事前学習済みモデルをコピー
        self.base_model = pretrained_model
        
        # 最終層を除く全ての層をフリーズするかどうか
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # MLPの場合、最終層を抽出して置き換え
        if isinstance(pretrained_model, SimpleMLP):
            # SimpleMLP用の処理
            # モデル内の最終層を新しいターゲットクラス数用に置き換え
            self.base_model.model[-1] = nn.Linear(self.base_model.model[-3].out_features, num_classes)
        else:
            # デフォルトの処理（BioKANモデル用）
            # BioKANモデルは通常classifierという属性を持つため
            try:
                in_features = self.base_model.classifier.in_features
                self.base_model.classifier = nn.Linear(in_features, num_classes)
            except AttributeError:
                print("注意: モデルに.classifier属性がありません。代替手段で対応します。")
                # 代替手段：モデルの最後の層をリフレクションで探して置き換え
                last_layer = None
                for name, module in self.base_model.named_modules():
                    if isinstance(module, nn.Linear):
                        last_layer = module
                
                if last_layer is not None:
                    in_features = last_layer.in_features
                    setattr(self.base_model, name, nn.Linear(in_features, num_classes))
    
    def forward(self, x):
        return self.base_model(x)

# データセットのロード
def load_datasets(batch_size=64, dataset_type='fashion_mnist'):
    print(f"\n{dataset_type}データセットを読み込み中...")
    
    # 変換の定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 標準的な正規化パラメータ
    ])
    
    # データセットの選択
    if dataset_type.lower() == 'fashion_mnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, transform=transform, download=True
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, transform=transform, download=True
        )
    elif dataset_type.lower() == 'cifar10':
        # CIFAR-10用の変換（カラー画像、サイズが異なる）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Grayscale(),  # グレースケールに変換（MNISTと合わせる）
            transforms.Resize((28, 28))  # MNISTと同じサイズに変更
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, transform=transform, download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, transform=transform, download=True
        )
    else:
        # デフォルトはFashion-MNIST
        print(f"警告: 未知のデータセットタイプ '{dataset_type}'。Fashion-MNISTを使用します。")
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, transform=transform, download=True
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, transform=transform, download=True
        )
    
    # データローダーの作成
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    
    print(f"訓練データサイズ: {len(train_dataset):,}件")
    print(f"テストデータサイズ: {len(test_dataset):,}件")
    
    return train_loader, test_loader

# 事前学習済みモデルのロード
def load_pretrained_model(model_path, model_type='biokan'):
    print(f"\n事前学習済みモデルをロード中: {model_path}")
    
    # MNISTの入力サイズ: 28x28 = 784
    in_features = 28 * 28
    
    if model_type.lower() == 'biokan' and BIOKAN_AVAILABLE:
        try:
            # BioKANモデルを使用
            model = create_biokan_classifier(
                in_features=in_features,
                hidden_dim=128,
                num_classes=10,  # MNIST: 10クラス (0-9)
                num_blocks=3,
                attention_type='standard',
                dropout=0.2,
                neuromodulation=True
            )
            # 保存されたパラメータをロード
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("BioKANモデルをロードしました")
            return model
        except Exception as e:
            print(f"BioKANモデルロードエラー: {str(e)}")
            print("標準的なMLPモデルにフォールバック...")
    
    # BioKANが使用できないか、エラーが発生した場合はシンプルなMLPを使用
    model = SimpleMLP(in_features, 128, 10)
    try:
        # 保存されたパラメータをロード
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("MLPモデルをロードしました")
    except Exception as e:
        print(f"モデルロードエラー: {str(e)}")
        print("ランダム初期化されたモデルを使用します")
    
    return model

# 転移学習の実行
def train_transfer_model(model, train_loader, test_loader, epochs=10, lr=0.0001, weight_decay=1e-5):
    print("\n転移学習を開始...")
    
    # 損失関数と最適化アルゴリズムの定義
    criterion = nn.CrossEntropyLoss()
    
    # 学習率を小さく設定（微調整のため）
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 学習履歴を保存
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'best_test_acc': 0.0,
        'best_epoch': 0
    }
    
    # 開始時間
    start_time = time.time()
    
    # エポックごとの学習ループ
    for epoch in range(epochs):
        model.train()  # 学習モード
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # データローダーからバッチを取得
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"エポック {epoch+1}/{epochs}")):
            # 画像とラベルをデバイスに転送
            images = images.reshape(images.shape[0], -1).to(device)  # [batch, 784]
            labels = labels.to(device)
            
            # 勾配をゼロに初期化
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(images)
            
            # 損失を計算
            loss = criterion(outputs, labels)
            
            # 逆伝播
            loss.backward()
            
            # パラメータを更新
            optimizer.step()
            
            # 統計情報を更新
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()
            
            # GPUメモリをクリア（オプション）
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # エポックの平均損失と精度を計算
        epoch_loss /= len(train_loader)
        epoch_acc = epoch_correct / epoch_total
        
        # テストデータでの評価
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        
        # 学習率を調整
        scheduler.step(test_loss)
        
        # 学習履歴を更新
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 最高精度を保存
        if test_acc > history['best_test_acc']:
            history['best_test_acc'] = test_acc
            history['best_epoch'] = epoch + 1
            
            # 最高精度のモデルを保存
            torch.save(model.state_dict(), 'best_transfer_model.pth')
            print(f"🔔 新しい最高精度を達成！モデルを保存しました")
        
        # エポックの結果を表示
        print(f"エポック {epoch+1}/{epochs} - "
              f"訓練損失: {epoch_loss:.4f}, 訓練精度: {epoch_acc:.4f}, "
              f"テスト損失: {test_loss:.4f}, テスト精度: {test_acc:.4f}")
    
    # 経過時間
    elapsed_time = time.time() - start_time
    print(f"\n転移学習完了！総時間: {elapsed_time:.1f}秒")
    print(f"最高テスト精度: {history['best_test_acc']:.4f} (エポック {history['best_epoch']})")
    
    # 最終モデルを保存
    torch.save(model.state_dict(), 'final_transfer_model.pth')
    print("最終モデルを保存しました: final_transfer_model.pth")
    
    return model, history

# モデルの評価
def evaluate_model(model, test_loader, criterion=None):
    model.eval()  # 評価モード
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(images.shape[0], -1).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    
    return test_loss, test_acc

# 学習履歴の可視化
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # 損失のプロット
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='訓練損失')
    plt.plot(history['test_loss'], label='テスト損失')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.title('転移学習 - 損失の推移')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 精度のプロット
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='訓練精度')
    plt.plot(history['test_acc'], label='テスト精度')
    plt.xlabel('エポック')
    plt.ylabel('精度')
    plt.title('転移学習 - 精度の推移')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_history.png')
    print("学習履歴グラフを保存しました: transfer_learning_history.png")

# 混同行列の可視化
def plot_confusion_matrix(model, test_loader, class_names):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(images.shape[0], -1).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # numpy配列に変換
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 混同行列の計算
    cm = confusion_matrix(all_labels, all_preds)
    
    # 混同行列の可視化
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("混同行列")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("予測ラベル")
    plt.ylabel("実際のラベル")
    
    # 各セルの値を表示
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('transfer_learning_confusion_matrix.png')
    print("混同行列を保存しました: transfer_learning_confusion_matrix.png")
    
    # クラスごとの精度
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(class_names)), class_accuracy)
    plt.xlabel("クラス")
    plt.ylabel("精度")
    plt.title("クラスごとの精度")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('transfer_learning_class_accuracy.png')
    print("クラス別精度を保存しました: transfer_learning_class_accuracy.png")

# サンプル画像と予測結果の可視化
def visualize_predictions(model, test_loader, class_names, num_samples=10):
    model.eval()
    
    # テストセットからサンプルを取得
    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # 予測を実行
    with torch.no_grad():
        images_flat = images.reshape(images.shape[0], -1).to(device)
        outputs = model(images_flat)
        _, predicted = torch.max(outputs, 1)
        
        # 予測確率を取得
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    # 画像と予測結果を表示
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # 画像を表示
        axes[i].imshow(images[i].squeeze().numpy(), cmap='gray')
        
        # 予測が正しいかどうかで色分け
        color = 'green' if predicted[i] == labels[i] else 'red'
        
        # タイトルに予測クラスと確率を表示
        pred_class = class_names[predicted[i].item()]
        true_class = class_names[labels[i].item()]
        confidence = probs[i, predicted[i]].item()
        
        axes[i].set_title(f"予測: {pred_class}\n実際: {true_class}\n確信度: {confidence:.2f}", 
                          color=color, fontsize=9)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('transfer_learning_predictions.png')
    print("予測結果のサンプルを保存しました: transfer_learning_predictions.png")

# メイン処理
if __name__ == "__main__":
    try:
        # データセットの選択
        target_dataset = 'fashion_mnist'  # 'fashion_mnist' または 'cifar10'
        
        # データのロード
        train_loader, test_loader = load_datasets(batch_size=128, dataset_type=target_dataset)
        
        # クラス名のマッピング
        if target_dataset.lower() == 'fashion_mnist':
            class_names = [fashion_labels[i] for i in range(10)]
        elif target_dataset.lower() == 'cifar10':
            # CIFAR-10のクラス名
            class_names = ['飛行機', '自動車', '鳥', '猫', '鹿', '犬', 'カエル', '馬', '船', 'トラック']
        else:
            class_names = [str(i) for i in range(10)]
        
        # 事前学習済みモデルのパス
        model_path = 'best_biokan_mnist_model.pth'
        
        # 事前学習済みモデルのロード
        if not os.path.exists(model_path):
            print(f"警告: 指定されたモデルファイル '{model_path}' が見つかりません。")
            print("ランダム初期化されたモデルを使用します。")
            
            # ランダムに初期化されたモデルを作成
            in_features = 28 * 28
            if BIOKAN_AVAILABLE:
                try:
                    base_model = create_biokan_classifier(
                        in_features=in_features,
                        hidden_dim=128,
                        num_classes=10,
                        num_blocks=3,
                        attention_type='standard',
                        dropout=0.2,
                        neuromodulation=True
                    )
                except Exception as e:
                    print(f"BioKANモデル初期化エラー: {str(e)}")
                    base_model = SimpleMLP(in_features, 128, 10)
            else:
                base_model = SimpleMLP(in_features, 128, 10)
        else:
            # 事前学習済みモデルをロード
            base_model = load_pretrained_model(model_path)
        
        # 転移学習モデルの作成
        transfer_model = TransferMLP(base_model, num_classes=10, freeze_base=False)
        transfer_model = transfer_model.to(device)
        print(f"転移学習モデルを{device}に転送しました")
        
        # 転移学習の実行
        trained_model, history = train_transfer_model(
            transfer_model,
            train_loader,
            test_loader,
            epochs=10,
            lr=0.0001,
            weight_decay=1e-5
        )
        
        # 学習履歴の可視化
        plot_training_history(history)
        
        # 混同行列と精度の可視化
        plot_confusion_matrix(trained_model, test_loader, class_names)
        
        # サンプル予測の可視化
        visualize_predictions(trained_model, test_loader, class_names)
        
        # 最終評価
        _, final_acc = evaluate_model(trained_model, test_loader)
        print(f"\n最終テスト精度: {final_acc:.4f}")
        
        print("\n転移学習したモデルは以下のパスに保存されています:")
        print("- 最高精度モデル: best_transfer_model.pth")
        print("- 最終モデル: final_transfer_model.pth")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        # GPUメモリ使用状況の表示
        if torch.cuda.is_available():
            print("\nGPUメモリ使用状況:")
            print(f"割り当て済みメモリ: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"キャッシュされたメモリ: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
            # エラーの詳細情報を出力
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()  # GPUメモリの解放
    
    finally:
        # クリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 