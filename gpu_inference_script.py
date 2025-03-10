"""
BioKANモデルのGPU推論スクリプト
事前学習済みモデルや転移学習済みモデルを使用してGPUで高速に推論を実行します
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os
from PIL import Image

# biokan_training.pyからモデルをインポート
from biokan_training import EnhancedBioKANModel, DynamicNeuromodulatorSystem
# biokan_transfer_learning.pyから転移学習モデルをインポート
from biokan_transfer_learning import TransferBioKANModel, get_dataset

# ===============================================
# GPU推論の実行関数
# ===============================================
def run_gpu_inference(model, data_loader, num_samples=10, task_type='classification'):
    """
    GPUを使用して高速に推論を実行します
    
    Args:
        model: 推論に使用するモデル
        data_loader: データローダー
        num_samples: 推論を実行するサンプル数
        task_type: タスクの種類
        
    Returns:
        推論結果のリスト
    """
    # GPUの確認
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPUを使用: {torch.cuda.get_device_name(0)}")
        
        # GPUキャッシュのクリア
        torch.cuda.empty_cache()
        
        # CUDNNベンチマークモードを有効化
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("警告: GPUが利用できないため、CPUを使用します")
    
    # モデルをGPUに転送
    model = model.to(device)
    model.eval()
    
    # 結果を格納するリスト
    results = []
    
    # 推論開始時間
    start_time = time.time()
    
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if i >= num_samples:
                break
            
            # データをGPUに転送
            data = data.to(device)
            
            # 混合精度推論
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    if hasattr(model, 'explain_prediction'):
                        # 転移学習モデルの場合
                        explanation = model.explain_prediction(data)
                        output = explanation['prediction']
                        
                        if task_type == 'classification':
                            # クラス予測
                            prediction = output[0]
                            confidence = explanation['confidence']
                        else:
                            # 回帰予測
                            prediction = output[0][0]
                            confidence = None
                        
                        # 神経伝達物質レベル
                        nt_levels = explanation['neurotransmitter_levels']
                    else:
                        # 通常のBioKANモデルの場合
                        output = model(data)
                        
                        if task_type == 'classification':
                            # クラス予測
                            prediction = torch.argmax(output, dim=1)[0].item()
                            probs = F.softmax(output, dim=1)
                            confidence = probs[0, prediction].item()
                        else:
                            # 回帰予測
                            prediction = output[0].item()
                            confidence = None
                        
                        # 神経伝達物質レベル
                        nt_levels = model.get_neuromodulator_levels() if hasattr(model, 'get_neuromodulator_levels') else {}
            else:
                # CPU推論の場合
                if hasattr(model, 'explain_prediction'):
                    # 転移学習モデルの場合
                    explanation = model.explain_prediction(data)
                    output = explanation['prediction']
                    
                    if task_type == 'classification':
                        # クラス予測
                        prediction = output[0]
                        confidence = explanation['confidence']
                    else:
                        # 回帰予測
                        prediction = output[0][0]
                        confidence = None
                    
                    # 神経伝達物質レベル
                    nt_levels = explanation['neurotransmitter_levels']
                else:
                    # 通常のBioKANモデルの場合
                    output = model(data)
                    
                    if task_type == 'classification':
                        # クラス予測
                        prediction = torch.argmax(output, dim=1)[0].item()
                        probs = F.softmax(output, dim=1)
                        confidence = probs[0, prediction].item()
                    else:
                        # 回帰予測
                        prediction = output[0].item()
                        confidence = None
                    
                    # 神経伝達物質レベル
                    nt_levels = model.get_neuromodulator_levels() if hasattr(model, 'get_neuromodulator_levels') else {}
            
            # 結果を保存
            result = {
                'data': data.cpu().numpy(),
                'target': target.item(),
                'prediction': prediction,
                'confidence': confidence,
                'neurotransmitter_levels': nt_levels
            }
            
            results.append(result)
            
            # 進捗表示
            print(f"サンプル {i+1}/{num_samples} の推論完了")
            if task_type == 'classification' and confidence is not None:
                print(f"  予測: クラス {prediction} (実際: {target.item()}) - 信頼度: {confidence*100:.2f}%")
            else:
                print(f"  予測: {prediction:.4f} (実際: {target.item()})")
    
    # 推論終了時間
    end_time = time.time()
    inference_time = end_time - start_time
    
    # パフォーマンス情報
    print(f"\n推論完了: {num_samples}サンプル")
    print(f"合計時間: {inference_time:.2f}秒")
    print(f"サンプルあたりの時間: {inference_time/num_samples*1000:.2f}ミリ秒")
    
    if device.type == 'cuda':
        print(f"GPU最大メモリ使用量: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    
    # 神経伝達物質レベルの可視化
    if len(results) > 0 and 'neurotransmitter_levels' in results[0] and results[0]['neurotransmitter_levels']:
        visualize_neurotransmitter_levels(results)
    
    return results

def visualize_neurotransmitter_levels(results):
    """
    神経伝達物質レベルの可視化
    
    Args:
        results: 推論結果のリスト
    """
    plt.figure(figsize=(12, 6))
    
    # サンプリングしたサンプル数
    num_samples = min(5, len(results))
    
    # サンプルごとの神経伝達物質レベル
    for i in range(num_samples):
        result = results[i]
        nt_levels = result['neurotransmitter_levels']
        
        # キーと値を取得
        keys = list(nt_levels.keys())
        values = [nt_levels[k] for k in keys]
        
        # 予測情報
        prediction = result['prediction']
        target = result['target']
        confidence = result['confidence']
        
        # サブプロット
        plt.subplot(1, num_samples, i+1)
        bars = plt.bar(keys, values)
        
        # バーの色分け
        for j, bar in enumerate(bars):
            # 値に応じて色を調整（活性度合いに応じた色）
            val = values[j]
            if keys[j] == 'gaba':  # GABAは抑制性
                color = plt.cm.Blues(abs(val/1.0))
            else:  # その他は興奮性
                color = plt.cm.Reds(abs(val/1.0))
            bar.set_color(color)
        
        plt.title(f"予測: {prediction}, 実際: {target}")
        plt.xticks(rotation=90)
        plt.ylim(-1.0, 1.0)
        plt.tight_layout()
    
    plt.suptitle("推論時の神経伝達物質レベル", fontsize=16)
    plt.savefig('neurotransmitter_inference_comparison.png')
    plt.close()

def visualize_sample_images(results, dataset_name):
    """
    サンプル画像の可視化
    
    Args:
        results: 推論結果のリスト
        dataset_name: データセット名
    """
    plt.figure(figsize=(15, 6))
    
    # サンプリングしたサンプル数
    num_samples = min(5, len(results))
    
    # クラス名の取得
    class_names = get_class_names(dataset_name)
    
    for i in range(num_samples):
        result = results[i]
        data = result['data'][0]  # バッチの最初のサンプル
        
        # 画像データを正規化から元に戻す
        if dataset_name == 'mnist' or dataset_name == 'fashion_mnist':
            # MNISTとFashion-MNISTは単一チャネル
            img = data[0]
            if dataset_name == 'mnist':
                img = img * 0.3081 + 0.1307  # MNISTの正規化を元に戻す
            else:
                img = img * 0.3530 + 0.2860  # Fashion-MNISTの正規化を元に戻す
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img, cmap='gray')
        else:
            # CIFAR-10は3チャネル
            img = np.transpose(data, (1, 2, 0))
            img = img * 0.5 + 0.5  # CIFAR-10の正規化を元に戻す
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img)
        
        # 予測情報
        prediction = result['prediction']
        target = result['target']
        confidence = result['confidence']
        
        pred_class = class_names[prediction] if class_names and prediction < len(class_names) else str(prediction)
        true_class = class_names[target] if class_names and target < len(class_names) else str(target)
        
        plt.title(f"予測: {pred_class}\n実際: {true_class}\n信頼度: {confidence*100:.1f}%")
        plt.axis('off')
    
    plt.suptitle(f"{dataset_name}の推論サンプル", fontsize=16)
    plt.savefig('inference_samples.png')
    plt.close()

def get_class_names(dataset_name):
    """
    データセットのクラス名を取得
    
    Args:
        dataset_name: データセット名
        
    Returns:
        クラス名のリスト
    """
    if dataset_name == 'mnist':
        return [str(i) for i in range(10)]
    elif dataset_name == 'fashion_mnist':
        return ['Tシャツ/トップ', 'ズボン', 'プルオーバー', 'ドレス', 'コート', 
                'サンダル', 'シャツ', 'スニーカー', 'バッグ', 'アンクルブーツ']
    elif dataset_name == 'cifar10':
        return ['飛行機', '自動車', '鳥', '猫', '鹿', 
                '犬', 'カエル', '馬', '船', 'トラック']
    else:
        return None

def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='BioKANモデルのGPU推論スクリプト')
    parser.add_argument('--model_type', type=str, choices=['biokan', 'transfer'], default='transfer',
                        help='使用するモデルタイプ (biokan: 通常のBioKANモデル, transfer: 転移学習モデル)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='モデルファイルのパス')
    parser.add_argument('--pretrained_model', type=str,
                        help='転移学習モデルを使用する場合の事前学習済みモデルのパス')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'cifar10', 'fashion_mnist'],
                        help='使用するデータセット')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='タスクの種類')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='推論を実行するサンプル数')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='バッチサイズ（通常は1）')
    parser.add_argument('--image_path', type=str,
                        help='単一画像ファイルのパス（オプション）')
    args = parser.parse_args()
    
    try:
        # GPU情報の表示
        if torch.cuda.is_available():
            print(f"GPU情報:")
            print(f"  デバイス名: {torch.cuda.get_device_name(0)}")
            print(f"  合計メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"  CUDA バージョン: {torch.version.cuda}")
            print(f"  PyTorch バージョン: {torch.__version__}")
        else:
            print("GPUが利用できません。CPUを使用します。")
        
        # モデルの読み込み
        if args.model_type == 'biokan':
            # 通常のBioKANモデルの場合
            model = EnhancedBioKANModel()
            model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
            print(f"BioKANモデルを読み込みました: {args.model_path}")
        else:
            # 転移学習モデルの場合
            if not args.pretrained_model:
                raise ValueError("転移学習モデルを使用する場合は--pretrained_modelを指定してください")
            
            # 事前学習済みモデルの読み込み
            base_model = EnhancedBioKANModel()
            base_model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
            print(f"事前学習済みモデルを読み込みました: {args.pretrained_model}")
            
            # 転移学習モデルの作成
            model = TransferBioKANModel(
                pretrained_model=base_model,
                task_type=args.task_type,
                num_classes=10,  # MNISTとFashion-MNISTは10クラス
                freeze_layers=True
            )
            
            # 転移学習済みモデルの重みをロード
            model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
            print(f"転移学習モデルを読み込みました: {args.model_path}")
        
        # 単一画像の推論
        if args.image_path:
            # 画像の読み込みと前処理
            image = Image.open(args.image_path).convert('RGB')
            
            # データセットに応じた前処理
            if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
                # グレースケールに変換
                image = image.convert('L')
                
                if args.dataset == 'mnist':
                    transform = transforms.Compose([
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.2860,), (0.3530,))
                    ])
            else:
                # CIFAR-10の場合
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            
            # 画像の前処理
            img_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加
            
            # 推論の実行
            print(f"\n画像ファイル {args.image_path} の推論を実行...")
            
            # GPUで推論
            if torch.cuda.is_available():
                device = torch.device('cuda')
                model = model.to(device)
                img_tensor = img_tensor.to(device)
                
                with torch.no_grad():
                    # 混合精度推論
                    with torch.cuda.amp.autocast():
                        if hasattr(model, 'explain_prediction'):
                            # 転移学習モデルの場合
                            explanation = model.explain_prediction(img_tensor)
                            output = explanation['prediction']
                            
                            if args.task_type == 'classification':
                                prediction = output[0]
                                confidence = explanation['confidence']
                            else:
                                prediction = output[0][0]
                                confidence = None
                            
                            nt_levels = explanation['neurotransmitter_levels']
                        else:
                            # 通常のBioKANモデルの場合
                            output = model(img_tensor)
                            
                            if args.task_type == 'classification':
                                prediction = torch.argmax(output, dim=1)[0].item()
                                probs = F.softmax(output, dim=1)
                                confidence = probs[0, prediction].item()
                            else:
                                prediction = output[0].item()
                                confidence = None
                            
                            nt_levels = model.get_neuromodulator_levels()
            else:
                # CPUで推論
                model.eval()
                with torch.no_grad():
                    if hasattr(model, 'explain_prediction'):
                        # 転移学習モデルの場合
                        explanation = model.explain_prediction(img_tensor)
                        output = explanation['prediction']
                        
                        if args.task_type == 'classification':
                            prediction = output[0]
                            confidence = explanation['confidence']
                        else:
                            prediction = output[0][0]
                            confidence = None
                        
                        nt_levels = explanation['neurotransmitter_levels']
                    else:
                        # 通常のBioKANモデルの場合
                        output = model(img_tensor)
                        
                        if args.task_type == 'classification':
                            prediction = torch.argmax(output, dim=1)[0].item()
                            probs = F.softmax(output, dim=1)
                            confidence = probs[0, prediction].item()
                        else:
                            prediction = output[0].item()
                            confidence = None
                        
                        nt_levels = model.get_neuromodulator_levels()
            
            # クラス名の取得
            class_names = get_class_names(args.dataset)
            pred_class = class_names[prediction] if class_names and prediction < len(class_names) else str(prediction)
            
            # 結果の表示
            print(f"推論結果:")
            if args.task_type == 'classification' and confidence is not None:
                print(f"  予測クラス: {pred_class} (クラスID: {prediction})")
                print(f"  信頼度: {confidence*100:.2f}%")
            else:
                print(f"  予測値: {prediction:.4f}")
            
            # 神経伝達物質レベルの表示
            print("\n神経伝達物質レベル:")
            for nt, level in nt_levels.items():
                print(f"  {nt}: {level:.4f}")
            
            # 可視化
            plt.figure(figsize=(10, 8))
            
            # 元画像
            plt.subplot(2, 1, 1)
            if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image)
            
            title = f"予測: {pred_class}"
            if confidence is not None:
                title += f" (信頼度: {confidence*100:.2f}%)"
            plt.title(title)
            plt.axis('off')
            
            # 神経伝達物質レベル
            plt.subplot(2, 1, 2)
            bars = plt.bar(nt_levels.keys(), nt_levels.values())
            
            # バーの色分け
            for j, bar in enumerate(bars):
                key = list(nt_levels.keys())[j]
                val = nt_levels[key]
                if key == 'gaba':  # GABAは抑制性
                    color = plt.cm.Blues(abs(val/1.0))
                else:  # その他は興奮性
                    color = plt.cm.Reds(abs(val/1.0))
                bar.set_color(color)
            
            plt.title("神経伝達物質レベル")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig('single_image_inference.png')
            plt.close()
            
            print(f"\n推論結果を single_image_inference.png に保存しました")
            return
        
        # データセットの取得
        _, test_dataset = get_dataset(args.dataset)
        
        # データローダーの作成
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        # 推論の実行
        print(f"\nデータセット {args.dataset} で {args.num_samples} サンプルの推論を実行...")
        results = run_gpu_inference(model, test_loader, args.num_samples, args.task_type)
        
        # サンプル画像の可視化
        visualize_sample_images(results, args.dataset)
        print(f"\n推論サンプルを inference_samples.png に保存しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 