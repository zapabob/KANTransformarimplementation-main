"""
BioKANモデルを使用した転移学習と推論問題のためのスクリプト
事前学習済みモデルをファインチューニングして、様々な推論タスクに適用します
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import argparse
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm.auto import tqdm  # tqdmをインポート
import optuna

# biokan_training.pyからモデルをインポート
from biokan_training import EnhancedBioKANModel, DynamicNeuromodulatorSystem, BiologicalAttention
from cuda_info_manager import print_cuda_info, get_device, setup_japanese_fonts

# 日本語フォントの設定（詳細表示しない）
setup_japanese_fonts(verbose=False)

# グローバルでdeviceを定義（他の関数から参照するため）
device = get_device()

# ===============================================
# 転移学習用のモデル拡張
# ===============================================
class TransferBioKANModel(nn.Module):
    """
    転移学習用に拡張されたBioKANモデル
    - 事前学習済みの重みを利用
    - 新しいタスクのための出力層を追加
    - 様々な推論タスク（分類、回帰、時系列など）に対応
    """
    def __init__(self, 
                 pretrained_model: EnhancedBioKANModel,
                 task_type: str = 'classification',
                 num_classes: int = 10, 
                 output_dim: int = 1,
                 freeze_layers: bool = True,
                 additional_params: dict = None):
        """
        Args:
            pretrained_model: 事前学習済みのBioKANモデル
            task_type: タスクの種類 ('classification', 'regression', 'sequence', 
                      'segmentation', 'anomaly_detection', 'multivariate_regression')
            num_classes: 分類/セグメンテーションタスクの場合のクラス数
            output_dim: 回帰/系列タスクの場合の出力次元
            freeze_layers: 事前学習済み層を凍結するかどうか
            additional_params: 追加のタスク固有パラメータ
        """
        super(TransferBioKANModel, self).__init__()
        
        # 事前学習済みモデル
        self.pretrained_model = pretrained_model
        
        # タスクの種類
        self.task_type = task_type
        
        # 事前学習済み層を凍結するかどうかのフラグ
        self.freeze_pretrained = freeze_layers
        
        # 追加パラメータの保存
        self.additional_params = additional_params or {}
        
        # 事前学習済み層の凍結（オプション）
        if freeze_layers:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
                
        # 元のモデルの隠れ層次元を取得
        hidden_dim = self.pretrained_model.hidden_dim
        
        # タスクに応じた出力層
        if task_type == 'classification':
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        elif task_type == 'regression':
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        elif task_type == 'multivariate_regression':
            # 多変量回帰用の出力層
            output_dim = self.additional_params.get('output_dim', 3)  # デフォルトは3次元出力
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        elif task_type == 'sequence':
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
            self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        elif task_type == 'segmentation':
            # 画像セグメンテーション用の出力層
            # UNet風のデコーダー構造
            img_size = self.additional_params.get('img_size', 28)  # デフォルトはMNISTサイズ
            
            # エンコード済みの特徴をデコードするための逆畳み込み層
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(hidden_dim // 4),
                nn.Conv2d(hidden_dim // 4, num_classes, kernel_size=1)
            )
            
            # 特徴マップをデコーダーに渡すための形状変換
            self.feature_reshape = nn.Linear(hidden_dim, hidden_dim * (img_size // 4) * (img_size // 4))
        elif task_type == 'anomaly_detection':
            # 異常検知のための自己符号化器構造
            self.encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4)
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim // 4, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
            # 異常スコア計算のための閾値（訓練中に更新）
            self.anomaly_threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        else:
            raise ValueError(f"サポートされていないタスク種類: {task_type}")
        
        # 予測履歴（説明可能性のため）
        self.prediction_history = []
        
        # 特徴重要度
        self.feature_importance = None
    
    def forward(self, x, return_features=False):
        """
        順伝播処理
        
        Args:
            x: 入力テンソル
            return_features: 特徴量を返すかどうか
            
        Returns:
            モデルの出力、またはreturn_featuresがTrueの場合は(出力, 特徴量)のタプル
        """
        # デバッグ情報 - 静かモードで無効化
        verbose = False
        if verbose:
            print(f"TransferBioKANModel 入力形状: {x.shape}, タイプ: {type(x)}")
            
        # 入力形状の確認と修正
        # batch_size = x.size(0)  # 未使用変数を削除
        current_state = x
            
        # MNISTデータの場合（B, 1, 28, 28）から（B, 784）に変換
        if len(x.shape) == 4 and x.shape[1] == 1 and x.shape[2] == 28 and x.shape[3] == 28:
            x = x.reshape(x.size(0), -1)
            if verbose:
                print(f"  形状を変換: {x.shape}")
                
        # 入力サイズを確認
        if x.shape[1] != 784 and (len(x.shape) != 3 or x.shape[2] != self.input_size):
            raise ValueError(f"入力サイズが不正です。expected=784, got={x.shape[1]}")
        
        # タスク固有の処理
        if self.task_type == 'segmentation':
            # 画像セグメンテーション処理
            # 入力画像から特徴を抽出
            with torch.set_grad_enabled(not self.freeze_pretrained):
                _, activations = self.pretrained_model(x.view(x.size(0), -1), return_activations=True)
                features = activations['block_2'].squeeze(1)  # [batch, hidden_dim]
            
            # 特徴をデコーダーに適した形状に変換
            img_size = self.additional_params.get('img_size', 28)
            features_2d = self.feature_reshape(features)
            features_2d = features_2d.view(x.size(0), -1, img_size // 4, img_size // 4)
        
            # デコーダーでセグメンテーションマスクを生成
            output = self.decoder(features_2d)  # [batch, num_classes, H, W]
            
            if return_features:
                return output, features
            return output
            
        elif self.task_type == 'anomaly_detection':
            # 異常検知処理
            with torch.set_grad_enabled(not self.freeze_pretrained):
                _, activations = self.pretrained_model(x, return_activations=True)
                features = activations['block_2'].squeeze(1)  # [batch, hidden_dim]
            
            # エンコーディング
            encoded = self.encoder(features)
        
            # デコーディング
            reconstructed = self.decoder(encoded)
            
            # 再構成誤差（異常スコア）
            reconstruction_error = F.mse_loss(reconstructed, features, reduction='none').mean(dim=1)
            
            if return_features:
                return reconstruction_error, features
            return reconstruction_error
            
        else:
            # 通常の入力処理（分類、回帰、多変量回帰）
            # 勾配情報の適切な処理
            with torch.set_grad_enabled(not self.freeze_pretrained):
                # 中間特徴量を取得
                _, activations = self.pretrained_model(x, return_activations=True)
                # 最終ブロックの活性化を使用
                features_raw = activations['block_2']
                # 形状調整 [batch, 1, hidden_dim] -> [batch, hidden_dim]
                features = features_raw.squeeze(1)
            
            # 新しいタスク用の出力層
            output = self.output_layer(features)
            
            if return_features:
                return output, features
            return output
    
    def predict(self, x, threshold=0.5):
        """
        予測を行い、適切な形式で返す
        
        Args:
            x: 入力データ
            threshold: 分類閾値（二値分類/異常検知の場合）
            
        Returns:
            タスクに応じた予測結果
        """
        self.eval()
        with torch.no_grad():
            if self.task_type == 'classification':
                outputs = self.forward(x)
                if outputs.shape[1] > 1:  # 多クラス分類
                    predictions = torch.argmax(outputs, dim=1)
                else:  # 二値分類
                    predictions = (torch.sigmoid(outputs) > threshold).float()
            
            elif self.task_type in ['regression', 'multivariate_regression']:
                predictions = self.forward(x)
            
            elif self.task_type == 'sequence':
                predictions = self.forward(x)
            
            elif self.task_type == 'segmentation':
                outputs = self.forward(x)
                predictions = torch.argmax(outputs, dim=1)  # クラスごとの最大値を取得
            
            elif self.task_type == 'anomaly_detection':
                reconstruction_error = self.forward(x)
                # 異常検知: 再構成誤差が閾値を超えるかどうか
                predictions = (reconstruction_error > self.anomaly_threshold).float()
                # 異常スコアも保存
                self.anomaly_scores = reconstruction_error.detach().cpu()
            
            # 予測履歴の保存（説明可能性のため）
            self.prediction_history.append(predictions.detach().cpu())
            
        return predictions
    
    def explain_prediction(self, x, target_class=None):
        """
        予測の説明可能性を提供
        
        Args:
            x: 入力データ
            target_class: 説明対象のクラス（分類タスクの場合）
            
        Returns:
            説明情報を含む辞書
        """
        self.eval()
        
        # 特徴量と出力を取得
        explanation = {
            'neurotransmitter_levels': self.pretrained_model.get_neuromodulator_levels()
        }
        
        if self.task_type == 'classification':
            outputs, features = self.forward(x, return_features=True)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities.gather(1, prediction.unsqueeze(1)).item()
            
            explanation.update({
                'prediction': prediction.detach().cpu().numpy(),
                'confidence': confidence,
                'features': features.detach().cpu().numpy(),
                'class_probabilities': probabilities.detach().cpu().numpy()
            })
            
        elif self.task_type in ['regression', 'multivariate_regression']:
            outputs, features = self.forward(x, return_features=True)
            explanation.update({
                'prediction': outputs.detach().cpu().numpy(),
                'features': features.detach().cpu().numpy(),
                'confidence': None
            })
            
        elif self.task_type == 'sequence':
            outputs, features = self.forward(x, return_features=True)
            explanation.update({
                'prediction': outputs.detach().cpu().numpy(),
                'features': features.detach().cpu().numpy(),
                'confidence': None,
                'sequence_features': features.detach().cpu().numpy()
            })
            
        elif self.task_type == 'segmentation':
            outputs, features = self.forward(x, return_features=True)
            segment_map = torch.argmax(outputs, dim=1)
            confidence_map = torch.max(F.softmax(outputs, dim=1), dim=1)[0]
            
            explanation.update({
                'prediction': segment_map.detach().cpu().numpy(),
                'features': features.detach().cpu().numpy(),
                'confidence': confidence_map.detach().cpu().numpy(),
                'segment_probabilities': F.softmax(outputs, dim=1).detach().cpu().numpy()
            })
            
        elif self.task_type == 'anomaly_detection':
            reconstruction_error, features = self.forward(x, return_features=True)
            is_anomaly = (reconstruction_error > self.anomaly_threshold).float()
            
            explanation.update({
                'prediction': is_anomaly.detach().cpu().numpy(),
                'features': features.detach().cpu().numpy(),
                'confidence': None,
                'anomaly_score': reconstruction_error.detach().cpu().numpy(),
                'threshold': self.anomaly_threshold.item()
            })
        
        return explanation

# ===============================================
# データセットとデータ変換
# ===============================================
def get_dataset(dataset_name: str, transform=None):
    """
    データセットの取得
    
    Args:
        dataset_name: データセット名
        transform: データ変換（オプション）
        
    Returns:
        訓練データセットとテストデータセット
    """
    if dataset_name == 'mnist':
        # MNISTデータセット
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    elif dataset_name == 'cifar10':
        # CIFAR-10データセット
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
    
    elif dataset_name == 'fashion_mnist':
        # Fashion-MNISTデータセット
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        
        train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
    
    else:
        raise ValueError(f"サポートされていないデータセット: {dataset_name}")
    
    return train_dataset, test_dataset

# ===============================================
# 転移学習と微調整
# ===============================================
def fine_tune_model(model, train_loader, val_loader, epochs=10, lr=0.0001, task_type='classification'):
    """
    転移学習モデルのファインチューニングを行う
    
    Args:
        model: 転移学習モデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        epochs: エポック数
        lr: 学習率
        task_type: タスクの種類
        
    Returns:
        (履歴, ファインチューニング済みモデル) のタプル
    """
    # デバイスの設定
    device = next(model.parameters()).device
    
    # モデルを訓練モードに設定
    model.train()
    
    # オプティマイザの設定
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # タスクに応じた損失関数
    if task_type in ['classification', 'segmentation']:
        criterion = nn.CrossEntropyLoss()
    elif task_type in ['regression', 'multivariate_regression', 'sequence']:
        criterion = nn.MSELoss()
    elif task_type == 'anomaly_detection':
        # 異常検知の場合、再構成誤差を最小化
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"サポートされていないタスク種類: {task_type}")
    
    # 混合精度訓練のためのスケーラー
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # 履歴を保存するための辞書
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 最良のモデルを追跡
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nトレーニングを開始します...")
    # エポック全体のプログレスバー
    pbar_epochs = tqdm(range(epochs), desc="総進捗", position=0)
    
    for epoch in pbar_epochs:
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # バッチごとのプログレスバー
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"訓練 Epoch {epoch+1}/{epochs}", leave=False, position=1)):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # タスク固有の入力整形
            if task_type in ['classification', 'regression', 'multivariate_regression', 'anomaly_detection']:
                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)
            
            optimizer.zero_grad()
            
            # CPUモードや、PyTorchのCPU版ではautocastを使わない
            use_amp = torch.cuda.is_available()
            
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    
                    # タスク固有の損失計算
                    if task_type == 'segmentation':
                        loss = criterion(outputs, targets.long())
                    elif task_type == 'anomaly_detection':
                        # 自己符号化器の場合、入力と再構成誤差の最小化
                        loss = outputs.mean()  # 再構成誤差を最小化
                    else:
                        loss = criterion(outputs, targets)
                
                # GradScalerを使用した逆伝播（GPUモードのみ）
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPUモードまたはautocast非対応の場合は通常の訓練を実行
                outputs = model(inputs)
                
                # タスク固有の損失計算
                if task_type == 'segmentation':
                    loss = criterion(outputs, targets.long())
                elif task_type == 'anomaly_detection':
                    # 自己符号化器の場合、入力と再構成誤差の最小化
                    loss = outputs.mean()  # 再構成誤差を最小化
                else:
                    loss = criterion(outputs, targets)
                
                # 通常の逆伝播
                loss.backward()
                optimizer.step()
            
            # 統計量の更新
            train_loss += loss.item() * inputs.size(0)
            
            if task_type in ['classification', 'segmentation']:
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
        
        # エポックごとの訓練損失
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            # 検証用プログレスバー
            for inputs, targets in tqdm(val_loader, desc=f"検証 Epoch {epoch+1}/{epochs}", leave=False, position=1):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # タスク固有の入力整形
                if task_type in ['classification', 'regression', 'multivariate_regression', 'anomaly_detection']:
                    if inputs.dim() > 2:
                        inputs = inputs.view(inputs.size(0), -1)
                
                # CPUモードや、PyTorchのCPU版ではautocastを使わない
                use_amp = torch.cuda.is_available()
                
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(inputs)
                        
                        # タスク固有の損失計算
                        if task_type == 'segmentation':
                            batch_loss = criterion(outputs, targets.long())
                        elif task_type == 'anomaly_detection':
                            batch_loss = outputs.mean()
                        else:
                            batch_loss = criterion(outputs, targets)
                else:
                    # CPUモードまたはautocast非対応の場合は通常の推論を実行
                    outputs = model(inputs)
                    
                    # タスク固有の損失計算
                    if task_type == 'segmentation':
                        batch_loss = criterion(outputs, targets.long())
                    elif task_type == 'anomaly_detection':
                        batch_loss = outputs.mean()
                    else:
                        batch_loss = criterion(outputs, targets)
                
                # 統計量の更新
                val_loss += batch_loss.item() * inputs.size(0)
                
                if task_type in ['classification', 'segmentation']:
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
        
        # エポックごとの検証損失
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # 結果の表示と進捗バーの更新
        pbar_epochs.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}', 
            'train_acc': f'{train_acc*100:.2f}%' if task_type in ['classification', 'segmentation'] else 'N/A',
            'val_acc': f'{val_acc*100:.2f}%' if task_type in ['classification', 'segmentation'] else 'N/A'
        })
        
        # 最良のモデルを追跡
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            pbar_epochs.set_description(f"総進捗 (最良: エポック{best_epoch+1})")
        
        # 履歴の更新
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
    
    print("\nトレーニングが完了しました")
    
    # 最良のモデルを復元
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, model

# ===============================================
# モデル評価と可視化
# ===============================================
def evaluate_model(model, test_loader, task_type='classification'):
    """
    転移学習モデルの評価
    
    Args:
        model: 転移学習モデル
        test_loader: テストデータローダー
        task_type: タスクの種類
        
    Returns:
        評価指標を含む辞書
    """
    # デバイスの設定
    device = next(model.parameters()).device
    
    # モデルを評価モードに設定
    model.eval()
    
    # 評価指標
    metrics = {}
    
    # 予測と正解のリスト
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        test_loss = 0.0
        
        if task_type == 'classification':
            test_correct = 0
            test_total = 0
            criterion = nn.CrossEntropyLoss()
        else:  # 回帰または系列
            criterion = nn.MSELoss()
        
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # データの形状整形（必要な場合）
            if inputs.dim() > 2 and task_type != 'sequence':
                inputs = inputs.view(inputs.size(0), -1)
            
            # 予測
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 統計更新
            test_loss += loss.item()
            
            if task_type == 'classification':
                # 分類タスクの評価
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                # 混同行列用に保存
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            else:
                # 回帰タスクの評価
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 平均損失
        test_loss /= len(test_loader)
        metrics['test_loss'] = test_loss
        
        if task_type == 'classification':
            # 分類精度
            test_acc = test_correct / test_total
            metrics['test_accuracy'] = test_acc
            
            # 混同行列
            cm = confusion_matrix(all_targets, all_predictions)
            metrics['confusion_matrix'] = cm
            
            print(f"テスト損失: {test_loss:.4f}")
            print(f"テスト精度: {test_acc:.4f}")
        else:
            # 回帰指標
            mse = mean_squared_error(all_targets, all_predictions)
            r2 = r2_score(all_targets, all_predictions)
            metrics['mse'] = mse
            metrics['r2'] = r2
            
            print(f"テスト損失: {test_loss:.4f}")
            print(f"平均二乗誤差: {mse:.4f}")
            print(f"決定係数(R²): {r2:.4f}")
    
    # 結果の可視化
    visualize_results(all_targets, all_predictions, task_type)
    
    return metrics

def visualize_results(targets, predictions, task_type='classification'):
    """
    評価結果の可視化
    
    Args:
        targets: 正解ラベル
        predictions: 予測
        task_type: タスクの種類
    """
    plt.figure(figsize=(12, 6))
    
    if task_type == 'classification':
        # 混同行列の可視化
        cm = confusion_matrix(targets, predictions)
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('予測クラス')
        plt.ylabel('実際のクラス')
        plt.title('混同行列')
        
        # クラスごとの精度
        plt.subplot(1, 2, 2)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        plt.bar(range(len(class_accuracy)), class_accuracy)
        plt.xlabel('クラス')
        plt.ylabel('精度')
        plt.title('クラスごとの精度')
        plt.xticks(range(len(class_accuracy)))
    else:
        # 回帰結果の可視化
        plt.scatter(targets, predictions, alpha=0.5)
        
        # 理想的な線（y=x）
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('実測値')
        plt.ylabel('予測値')
        plt.title('回帰予測の結果')
        
        # テキスト情報
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        plt.figtext(0.15, 0.8, f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
        fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    plt.tight_layout()
    plt.savefig(f'transfer_learning_results_{task_type}.png')
    plt.close()

# ===============================================
# 推論と説明
# ===============================================
def run_inference(model, data_sample, task_type='classification', threshold=0.5):
    """
    単一のデータサンプルに対して推論を実行し、結果を返す
    
    Args:
        model: 訓練済みモデル
        data_sample: 入力データサンプル
        task_type: タスクの種類
        threshold: 分類の閾値（デフォルト: 0.5）
        
    Returns:
        prediction: 予測結果
        explanation: 説明可能なAI出力（利用可能な場合）
    """
    model.eval()
    
    # データをデバイスに移動
    if isinstance(data_sample, torch.Tensor):
        data_sample = data_sample.to(device)
    else:
        data_sample = torch.FloatTensor(data_sample).to(device)
    
    # 入力整形
    if task_type in ['classification', 'regression'] and data_sample.dim() > 2:
        data_sample = data_sample.view(1, -1)  # バッチサイズ1を維持
    
    # 勾配計算なしで推論
    with torch.no_grad():
        # CPUモードやPyTorchのCPU版では通常の推論、GPUモードでは混合精度推論
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                # 予測と説明を取得
                explanation = model.explain_prediction(data_sample)
        else:
            # 通常の推論
            explanation = model.explain_prediction(data_sample)
    
    # 説明の整形（Noneでない場合）
    if explanation is not None:
        if isinstance(explanation, torch.Tensor):
            explanation = explanation.cpu().numpy()
        
        # 説明データの形式を調整（必要に応じて）
        if task_type == 'classification':
            # クラスごとの説明を提供
            pass
        elif task_type == 'segmentation':
            # セグメンテーション用の説明を整形
            explanation = explanation.reshape((28, 28))  # MNIST/Fashion-MNISTのサイズに調整
    
    # タスクに応じた予測を返す
    prediction = model(data_sample)
    
    # 信頼度を計算して説明に追加
    if task_type == 'classification' and isinstance(prediction, torch.Tensor):
        confidence = torch.softmax(prediction, dim=1).max().item()
        if explanation is None:
            explanation = {}
        if isinstance(explanation, dict):
            explanation['confidence'] = confidence
    
    if task_type == 'classification':
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        
        # クラスラベルに変換
        prediction = np.argmax(prediction, axis=1)
    
    elif task_type == 'regression':
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
    
    elif task_type == 'segmentation':
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        
        # セグメンテーションマスクに変換
        prediction = np.argmax(prediction, axis=1)
        prediction = prediction.reshape((28, 28))  # 画像の形状に戻す
    
    return prediction, explanation

# ===============================================
# メイン関数
# ===============================================
def main():
    """
    メイン関数：コマンドライン引数を解析し、転移学習とモデル評価を実行
    """
    parser = argparse.ArgumentParser(description='BioKANのMNIST転移学習')
    
    # 基本パラメータ
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='使用するデータセット')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression', 'multivariate_regression', 'sequence', 'segmentation', 'anomaly_detection'],
                        help='転移学習のタスク種類')
    parser.add_argument('--pretrained_model', type=str, default='biokan_trained_models/best_biokan_model.pth',
                        help='事前学習済みモデルのパス')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=5,
                        help='ファインチューニングのエポック数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学習率')
    parser.add_argument('--freeze', action='store_true',
                        help='事前学習済み層を凍結するかどうか')
    parser.add_argument('--use_gpu', action='store_true',
                        help='GPUを使用するかどうか')
    parser.add_argument('--inference_only', action='store_true',
                        help='ファインチューニングを行わず推論のみ実行')
    parser.add_argument('--inference_model', type=str,
                        help='推論に使用する事前学習済み転移学習モデルのパス')
    parser.add_argument('--debug', action='store_true',
                        help='デバッグモードを有効化')
    args = parser.parse_args()
    
    # GPUの確認
    if args.use_gpu:
        device = get_device()
        print_cuda_info(verbose=True)
    else:
        device = torch.device('cpu')
        print("CPUモードで実行します")
    
    print(f"\n{args.task_type}タスクの転移学習を開始します...")
    
    # 推論のみモードの場合
    if args.inference_only and args.inference_model:
        print(f"推論のみモード: モデル {args.inference_model} を使用")
        
        # データセットの取得（テスト用）
        _, test_dataset = get_dataset(args.dataset)
        
        # 転移学習モデルの読み込み
        try:
            # オリジナルのモデル構造を作成
            base_model = EnhancedBioKANModel()
            base_model.to(device)
            
            # 事前学習済みの重みをロード
            base_model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
            
            # 転移学習モデルを作成
            transfer_model = TransferBioKANModel(
                pretrained_model=base_model,
                task_type=args.task_type,
                num_classes=10,  # MNISTとFashion-MNISTは10クラス
                freeze_layers=True
            )
            
            # 転移学習済みモデルの重みをロード
            transfer_model.load_state_dict(torch.load(args.inference_model, map_location=device))
            transfer_model.to(device)
            
            print(f"転移学習済みモデルを読み込みました: {args.inference_model}")
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return
        
        # テストデータから数サンプル取得
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1,
            shuffle=True,
            pin_memory=device.type == 'cuda',
            num_workers=4 if device.type == 'cuda' else 0
        )
        
        # 5サンプルで推論を実行
        for i, (data, target) in enumerate(test_loader):
            if i >= 5:
                break
                
            print(f"\nサンプル {i+1} の推論実行...")
            prediction, explanation = run_inference(transfer_model, data, args.task_type, threshold=0.5)
            
            # 結果の表示
            if args.task_type == 'classification':
                predicted_class = prediction[0]
                confidence = explanation['confidence'] * 100
                print(f"推論結果: クラス {predicted_class} (実際のクラス: {target.item()}, 信頼度: {confidence:.2f}%)")
            else:
                predicted_value = prediction[0][0]
                print(f"推論結果: {predicted_value:.4f} (実際の値: {target.item()})")
            
            print("神経伝達物質レベル:")
            for nt, level in explanation['neurotransmitter_levels'].items():
                print(f"  {nt}: {level:.4f}")
        
        print("\n推論処理が完了しました。")
        return
    
    # 事前学習済みモデルのロード
    try:
        if os.path.exists(args.pretrained_model):
            # オリジナルのモデル構造を作成
            base_model = EnhancedBioKANModel()
            # 事前学習済みの重みをロード
            base_model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
            print(f"事前学習済みモデルを読み込みました: {args.pretrained_model}")
        else:
            raise FileNotFoundError(f"モデルファイルが見つかりません: {args.pretrained_model}")
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return
    
    # データセットの取得
    train_dataset, test_dataset = get_dataset(args.dataset)
    
    # データセットの分割（訓練・検証・テスト）
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_subset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=device.type == 'cuda',
        num_workers=4 if device.type == 'cuda' else 0
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=args.batch_size,
        pin_memory=device.type == 'cuda',
        num_workers=4 if device.type == 'cuda' else 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size if not args.use_gpu else 128,  # GPUの場合は大きなバッチサイズを使用
        pin_memory=device.type == 'cuda',
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # タスク設定
    num_classes = 10  # MNISTとFashion-MNISTは10クラス
    if args.dataset == 'cifar10':
        num_classes = 10  # CIFAR-10も10クラス
    
    # 転移学習モデルの作成
    transfer_model = TransferBioKANModel(
        pretrained_model=base_model,
        task_type=args.task_type,
        num_classes=num_classes,
        output_dim=1,  # 回帰タスクの場合の出力次元
        freeze_layers=args.freeze
    )
    transfer_model = transfer_model.to(device)
    
    # モデル情報の表示
    print(f"転移学習モデル構成:")
    print(f"  タスク種類: {args.task_type}")
    print(f"  データセット: {args.dataset}")
    print(f"  事前学習層凍結: {args.freeze}")
    print(f"  合計パラメータ数: {sum(p.numel() for p in transfer_model.parameters())}")
    print(f"  学習可能パラメータ数: {sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)}")
    
    # モデルの微調整
    print("\nモデルのファインチューニング開始...")
    history, tuned_model = fine_tune_model(
        model=transfer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        task_type=args.task_type
    )
    
    # 学習曲線のプロット
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='訓練損失')
    plt.plot(history['val_loss'], label='検証損失')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.legend()
    plt.title('学習曲線 - 損失')
    
    if args.task_type == 'classification':
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='訓練精度')
        plt.plot(history['val_acc'], label='検証精度')
        plt.xlabel('エポック')
        plt.ylabel('精度')
        plt.legend()
        plt.title('学習曲線 - 精度')
    
    plt.tight_layout()
    plt.savefig('transfer_learning_history.png')
    plt.close()
    
    # テストデータでの評価
    print("\nテストデータでの評価...")
    metrics = evaluate_model(tuned_model, test_loader, args.task_type)
    
    # モデルの保存
    save_path = f'transfer_model_{args.dataset}_{args.task_type}.pth'
    torch.save(tuned_model.state_dict(), save_path)
    print(f"モデルを保存しました: {save_path}")
    
    # サンプル推論
    print("\nサンプル推論の実行...")
    # テストデータから1サンプル取得
    sample_data, sample_target = next(iter(test_loader))
    sample_data = sample_data[0].unsqueeze(0)  # 単一サンプル
    
    # 推論と説明の実行
    prediction, explanation = run_inference(tuned_model, sample_data, args.task_type, threshold=0.5)
    
    # 結果の表示
    if args.task_type == 'classification':
        predicted_class = prediction[0]
        confidence = explanation['confidence'] * 100
        print(f"推論結果: クラス {predicted_class} (信頼度: {confidence:.2f}%)")
    else:
        predicted_value = prediction[0][0]
        print(f"推論結果: {predicted_value:.4f}")
    
    print("神経伝達物質レベル:")
    for nt, level in explanation['neurotransmitter_levels'].items():
        print(f"  {nt}: {level:.4f}")
    
    print("\n転移学習が完了しました。")

def optimize_hyperparameters(base_model, train_loader, val_loader, task_type='classification', n_trials=30, save_best_params=True):
    """
    Optunaを使用してハイパーパラメータを最適化する
    
    Args:
        base_model: 事前学習済みモデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        task_type: タスクの種類
        n_trials: 試行回数
        save_best_params: 最適なパラメータをJSONファイルに保存するかどうか
        
    Returns:
        最適なハイパーパラメータと最良のモデル
    """
    def objective(trial):
        # ハイパーパラメータの候補を定義
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'freeze_layers': trial.suggest_categorical('freeze_layers', [True, False]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 5, 50),
            'early_stopping_patience': trial.suggest_int('early_stopping_patience', 3, 10)
        }
        
        print(f"\nトライアル {trial.number}: パラメータ")
        for param_name, param_value in params.items():
            print(f"  {param_name}: {param_value}")
        
        # モデルの作成
        transfer_model = TransferBioKANModel(
            pretrained_model=base_model,
            task_type=task_type,
            num_classes=10 if task_type == 'classification' else 1,
            freeze_layers=params['freeze_layers']
        )
        
        transfer_model.to(device)
        
        # モデルのDropout率を更新
        for module in transfer_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = params['dropout']
        
        # オプティマイザ設定
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, transfer_model.parameters()),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # 損失関数
        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # 混合精度訓練のためのスケーラー
        scaler = GradScaler(enabled=torch.cuda.is_available())
        
        # 早期終了の設定
        patience = params['early_stopping_patience']
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # トレーニング
        transfer_model.train()
        
        # トライアルのプログレスバー
        pbar_epochs = tqdm(range(params['epochs']), 
                          desc=f"トライアル {trial.number}/{n_trials}", 
                          position=0)
        
        for epoch in pbar_epochs:
            # 訓練フェーズ
            transfer_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # バッチごとのプログレスバー
            train_iter = tqdm(train_loader, 
                             desc=f"訓練 Epoch {epoch+1}/{params['epochs']}", 
                             leave=False, 
                             position=1)
            
            for inputs, targets in train_iter:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 入力データの整形
                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)
                
                optimizer.zero_grad()
                
                # 混合精度訓練
                if torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = transfer_model(inputs)
                        loss = criterion(outputs, targets)
                    
                    # GradScalerを使用した逆伝播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = transfer_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                # 統計量の更新
                train_loss += loss.item() * inputs.size(0)
                
                if task_type == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
            
            train_loss /= len(train_loader.dataset)
            train_acc = train_correct / train_total if task_type == 'classification' else 0.0
            
            # 検証フェーズ
            transfer_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                # 検証用プログレスバー
                val_iter = tqdm(val_loader, 
                              desc=f"検証 Epoch {epoch+1}/{params['epochs']}", 
                              leave=False, 
                              position=1)
                
                for inputs, targets in val_iter:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # 入力データの整形
                    if inputs.dim() > 2:
                        inputs = inputs.view(inputs.size(0), -1)
                    
                    # 混合精度推論
                    if torch.cuda.is_available():
                        with torch.amp.autocast(device_type='cuda'):
                            outputs = transfer_model(inputs)
                            batch_loss = criterion(outputs, targets)
                    else:
                        outputs = transfer_model(inputs)
                        batch_loss = criterion(outputs, targets)
                    
                    val_loss += batch_loss.item() * inputs.size(0)
                    
                    if task_type == 'classification':
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += targets.size(0)
                        val_correct += (predicted == targets).sum().item()
            
            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / val_total if task_type == 'classification' else 0.0
            
            # 進捗バーの更新
            pbar_epochs.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}', 
                'train_acc': f'{train_acc*100:.2f}%' if task_type == 'classification' else 'N/A',
                'val_acc': f'{val_acc*100:.2f}%' if task_type == 'classification' else 'N/A'
            })
            
            # 早期終了の判定
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    pbar_epochs.set_description(f"トライアル {trial.number}/{n_trials} (早期終了: {epoch+1}エポック)")
                    break
            
            # 進捗を報告
            trial.report(val_loss, epoch)
            
            # プルーニング（性能が悪いトライアルの早期中断）
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_val_loss
    
    # Optunaの設定
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # 最適なハイパーパラメータを表示
    best_params = study.best_params
    print("\n最適なハイパーパラメータ:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    
    # 最適パラメータの保存
    if save_best_params:
        best_params_path = f'best_params_{task_type}_mnist.json'
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"最適パラメータを保存しました: {best_params_path}")
    
    # 可視化（オプション）
    try:
        # 最適化履歴
        optimization_history_plot = optuna.visualization.plot_optimization_history(study)
        param_importance_plot = optuna.visualization.plot_param_importances(study)
        
        # 保存
        optimization_history_plot.write_image("optuna_history.png")
        param_importance_plot.write_image("optuna_param_importance.png")
        print("最適化結果のプロットを保存しました")
    except Exception as e:
        print(f"プロットの保存に失敗しました: {e}")
    
    return best_params, study

if __name__ == "__main__":
    main() 