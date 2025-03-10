"""
CUDA 12対応のBioKANモデル推論スクリプト
CUDA 12環境で最適化された高速推論を実行します
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
import json
import sys
from PIL import Image

# バージョン情報
print(f"PyTorch バージョン: {torch.__version__}")
print(f"Python バージョン: {sys.version}")

# GPU/CUDA情報の確認
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"CUDA バージョン: {cuda_version}")
    
    # CUDA 12の互換性チェック
    if cuda_version.startswith('12.'):
        print("CUDA 12が検出されました。最適化された推論を行います。")
    else:
        print(f"警告: CUDA {cuda_version}が検出されました。このスクリプトはCUDA 12向けに最適化されています。")
    
    # GPU情報
    device_count = torch.cuda.device_count()
    print(f"利用可能なGPUデバイス数: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {device_name} (Compute Capability: {device_capability[0]}.{device_capability[1]})")
        
        # メモリ情報
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  合計メモリ: {total_memory:.2f} GB")
else:
    print("警告: GPUが検出されませんでした。CPUで実行します。")

try:
    # biokan_training.pyからモデルをインポート
    from biokan_training import EnhancedBioKANModel, DynamicNeuromodulatorSystem
    # biokan_transfer_learning.pyから転移学習モデルをインポート
    from biokan_transfer_learning import TransferBioKANModel, get_dataset
except ImportError as e:
    print(f"モジュールのインポートエラー: {e}")
    print("必要なモジュールがインストールされていることを確認してください。")
    sys.exit(1)

# ===============================================
# CUDA 12最適化設定
# ===============================================
def setup_cuda12_optimization():
    """CUDA 12向けの最適化設定を行います"""
    if torch.cuda.is_available():
        # CUDNNベンチマークモードを有効化（カーネル選択を最適化）
        torch.backends.cudnn.benchmark = True
        
        # TensorCoreの利用を有効化（可能な場合）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 確率的演算の決定性を確保（オプション）
        # torch.backends.cudnn.deterministic = True
        
        # キャッシュクリア
        torch.cuda.empty_cache()
        
        print("CUDA 12最適化設定が完了しました")
        print("  - CUDNNベンチマークモード: 有効")
        print("  - TensorCore (TF32): 有効（利用可能な場合）")
        # print("  - 確率的演算の決定性: 有効")
        
        return True
    else:
        print("GPUが利用できないため、CUDA 12最適化は適用されません")
        return False

# ===============================================
# CUDA 12対応の推論関数
# ===============================================
def run_cuda12_inference(model, data_loader, num_samples=10, task_type='classification', precision='mixed'):
    """
    CUDA 12を使用して高速に推論を実行します
    
    Args:
        model: 推論に使用するモデル
        data_loader: データローダー
        num_samples: 推論を実行するサンプル数
        task_type: タスクの種類
        precision: 計算精度 ('mixed', 'float32', 'float16', 'bfloat16')
        
    Returns:
        推論結果のリスト
    """
    # GPUの確認
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPUを使用: {torch.cuda.get_device_name(0)}")
        
        # CUDA 12最適化の設定
        setup_cuda12_optimization()
    else:
        device = torch.device('cpu')
        print("警告: GPUが利用できないため、CPUを使用します")
        # CPUの場合、精度設定をデフォルトに戻す
        precision = 'float32'
    
    # モデルをGPUに転送
    model = model.to(device)
    model.eval()
    
    # 結果を格納するリスト
    results = []
    
    # 精度設定の表示
    print(f"推論精度モード: {precision}")
    
    # 推論開始時間
    start_time = time.time()
    
    # 精度設定の選択
    if precision == 'float16' and device.type == 'cuda':
        # FP16の場合
        dtype = torch.float16
        autocast_enabled = False
        print("FP16精度で推論を実行")
        model = model.half()  # モデルをFP16に変換
    elif precision == 'bfloat16' and device.type == 'cuda' and torch.cuda.is_bf16_supported():
        # BF16の場合（一部のGPUのみサポート）
        dtype = torch.bfloat16
        autocast_enabled = False
        print("BF16精度で推論を実行")
        model = model.to(dtype=torch.bfloat16)  # モデルをBF16に変換
    elif precision == 'mixed' and device.type == 'cuda':
        # 混合精度の場合
        dtype = None  # autocastで自動的に処理
        autocast_enabled = True
        print("混合精度(AMP)で推論を実行")
    else:
        # デフォルトはFP32
        dtype = torch.float32
        autocast_enabled = False
        print("FP32精度で推論を実行")
    
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if i >= num_samples:
                break
            
            # データをGPUに転送
            data = data.to(device, dtype=dtype if dtype is not None else None)
            
            # 精度設定に応じた推論
            if autocast_enabled and device.type == 'cuda':
                # 混合精度推論
                with torch.cuda.amp.autocast():
                    # 推論実行
                    prediction, confidence, nt_levels = run_model_inference(model, data, task_type)
            else:
                # 通常精度推論
                prediction, confidence, nt_levels = run_model_inference(model, data, task_type)
            
            # 推論性能計測のためのGPU同期
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 結果を保存
            result = {
                'data': data.cpu().to(torch.float32).numpy(),  # 常にfloat32に戻す
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
        print(f"GPU割り当てキャッシュ: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    # 神経伝達物質レベルの可視化
    if len(results) > 0 and 'neurotransmitter_levels' in results[0] and results[0]['neurotransmitter_levels']:
        visualize_neurotransmitter_levels(results)
    
    return results

def run_model_inference(model, data, task_type='classification'):
    """
    モデル推論を実行し、予測結果を返します
    
    Args:
        model: 推論モデル
        data: 入力データ
        task_type: タスクの種類
        
    Returns:
        (予測, 信頼度, 神経伝達物質レベル)のタプル
    """
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
    
    return prediction, confidence, nt_levels

# ===============================================
# 可視化関数
# ===============================================
def visualize_neurotransmitter_levels(results):
    """
    神経伝達物質レベルの可視化
    
    Args:
        results: 推論結果のリスト
    """
    plt.figure(figsize=(14, 6))
    
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
    plt.savefig('cuda12_neurotransmitter_levels.png')
    plt.close()
    print("神経伝達物質レベルのグラフを cuda12_neurotransmitter_levels.png に保存しました")

def visualize_sample_images(results, dataset_name):
    """
    サンプル画像の可視化
    
    Args:
        results: 推論結果のリスト
        dataset_name: データセット名
    """
    plt.figure(figsize=(16, 6))
    
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
    
    plt.suptitle(f"{dataset_name}の推論サンプル (CUDA 12)", fontsize=16)
    plt.savefig('cuda12_inference_samples.png')
    plt.close()
    print("推論サンプルのグラフを cuda12_inference_samples.png に保存しました")

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

# ===============================================
# メイン関数
# ===============================================
def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='CUDA 12対応のBioKANモデル推論スクリプト')
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
    parser.add_argument('--precision', type=str, default='mixed',
                        choices=['mixed', 'float32', 'float16', 'bfloat16'],
                        help='計算精度 (mixed: 混合精度, float32: FP32, float16: FP16, bfloat16: BF16)')
    parser.add_argument('--image_path', type=str,
                        help='単一画像ファイルのパス（オプション）')
    parser.add_argument('--profile', action='store_true',
                        help='プロファイリングを有効化')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='結果の保存ディレクトリ')
    args = parser.parse_args()
    
    # 保存ディレクトリ作成
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # CUDA 12の最適化設定
        has_cuda = setup_cuda12_optimization()
        
        # プロファイリング設定
        if args.profile and has_cuda:
            print("\nプロファイリングを有効化...")
            profiler_schedule = torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1
            )
            profiler = torch.profiler.profile(
                schedule=profiler_schedule,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler.start()
        else:
            profiler = None
        
        # モデルの読み込み
        if args.model_type == 'biokan':
            # 通常のBioKANモデルの場合
            print("\nBioKANモデルを読み込み中...")
            model = EnhancedBioKANModel()
            
            # モデルファイルの存在確認
            if not os.path.exists(args.model_path):
                print(f"エラー: モデルファイル '{args.model_path}' が見つかりません")
                return
            
            model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
            print(f"BioKANモデルを読み込みました: {args.model_path}")
        else:
            # 転移学習モデルの場合
            if not args.pretrained_model:
                print("エラー: 転移学習モデルを使用する場合は--pretrained_modelを指定してください")
                return
            
            # 事前学習済みモデルの読み込み
            print("\n事前学習済みモデルを読み込み中...")
            
            # モデルファイルの存在確認
            if not os.path.exists(args.pretrained_model):
                print(f"エラー: 事前学習済みモデルファイル '{args.pretrained_model}' が見つかりません")
                return
            
            base_model = EnhancedBioKANModel()
            base_model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
            print(f"事前学習済みモデルを読み込みました: {args.pretrained_model}")
            
            # 転移学習モデルの作成
            print("転移学習モデルを作成中...")
            model = TransferBioKANModel(
                pretrained_model=base_model,
                task_type=args.task_type,
                num_classes=10,  # MNISTとFashion-MNISTは10クラス
                freeze_layers=True
            )
            
            # 転移学習モデルの読み込み
            # モデルファイルの存在確認
            if not os.path.exists(args.model_path):
                print(f"警告: 転移学習済みモデルファイル '{args.model_path}' が見つかりません")
                print("事前学習済みモデルのみを使用して推論を実行します")
            else:
                model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
                print(f"転移学習モデルを読み込みました: {args.model_path}")
        
        # 単一画像の推論
        if args.image_path:
            # 画像ファイルの存在確認
            if not os.path.exists(args.image_path):
                print(f"エラー: 画像ファイル '{args.image_path}' が見つかりません")
                return
            
            # 画像の読み込みと前処理
            print(f"\n画像ファイル {args.image_path} の推論を実行...")
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
            if torch.cuda.is_available():
                device = torch.device('cuda')
                model = model.to(device)
                img_tensor = img_tensor.to(device)
                
                # 精度設定
                if args.precision == 'float16':
                    model = model.half()
                    img_tensor = img_tensor.half()
                elif args.precision == 'bfloat16' and torch.cuda.is_bf16_supported():
                    model = model.to(dtype=torch.bfloat16)
                    img_tensor = img_tensor.to(dtype=torch.bfloat16)
                
                with torch.no_grad():
                    # 混合精度推論
                    if args.precision == 'mixed':
                        with torch.cuda.amp.autocast():
                            prediction, confidence, nt_levels = run_model_inference(model, img_tensor, args.task_type)
                    else:
                        prediction, confidence, nt_levels = run_model_inference(model, img_tensor, args.task_type)
            else:
                # CPUで推論
                model.eval()
                with torch.no_grad():
                    prediction, confidence, nt_levels = run_model_inference(model, img_tensor, args.task_type)
            
            # クラス名の取得
            class_names = get_class_names(args.dataset)
            pred_class = class_names[prediction] if class_names and prediction < len(class_names) else str(prediction)
            
            # 結果の表示
            print(f"\n推論結果:")
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
            
            plt.savefig(os.path.join(args.save_dir, 'cuda12_single_image_inference.png'))
            plt.close()
            
            print(f"\n推論結果を {os.path.join(args.save_dir, 'cuda12_single_image_inference.png')} に保存しました")
            
            # プロファイラーを停止
            if profiler:
                profiler.stop()
                profile_path = os.path.join(args.save_dir, 'cuda12_profile_single_image.json')
                profiler.export_chrome_trace(profile_path)
                print(f"プロファイル結果を {profile_path} に保存しました")
            
            return
        
        # データセットの取得
        print(f"\nデータセット {args.dataset} を読み込み中...")
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
        results = run_cuda12_inference(
            model, 
            test_loader, 
            args.num_samples, 
            args.task_type,
            args.precision
        )
        
        # サンプル画像の可視化
        visualize_sample_images(results, args.dataset)
        
        # プロファイラーを停止
        if profiler:
            profiler.stop()
            profile_path = os.path.join(args.save_dir, 'cuda12_profile.json')
            profiler.export_chrome_trace(profile_path)
            print(f"プロファイル結果を {profile_path} に保存しました")
        
        # 推論統計の出力
        json_stats = {
            'dataset': args.dataset,
            'task_type': args.task_type,
            'num_samples': args.num_samples,
            'precision': args.precision,
            'has_cuda': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            'results': []
        }
        
        for result in results:
            result_dict = {
                'target': int(result['target']),
                'prediction': int(result['prediction']) if args.task_type == 'classification' else float(result['prediction']),
                'confidence': float(result['confidence']) if result['confidence'] is not None else None,
                'neurotransmitter_levels': {k: float(v) for k, v in result['neurotransmitter_levels'].items()}
            }
            json_stats['results'].append(result_dict)
        
        with open(os.path.join(args.save_dir, 'cuda12_inference_results.json'), 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"推論結果の詳細を {os.path.join(args.save_dir, 'cuda12_inference_results.json')} に保存しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 