import torch
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# グローバル変数：CUDA情報が表示されたかを追跡
CUDA_INFO_DISPLAYED = False
# グローバル変数：フォント情報が表示されたかを追跡
FONT_INFO_DISPLAYED = False

# フラグファイルのパス
FLAG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cuda_info_displayed")
FONT_FLAG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".font_info_displayed")

def setup_japanese_fonts(verbose=False):
    """
    matplotlibで日本語フォントを使用するための設定
    
    Args:
        verbose (bool): フォント情報を表示するかどうか
    """
    global FONT_INFO_DISPLAYED
    
    # フラグファイルがあれば表示済みとみなす
    if os.path.exists(FONT_FLAG_FILE_PATH):
        FONT_INFO_DISPLAYED = True
        # フォント設定のみ行い、情報表示はスキップ
        plt.rcParams['font.family'] = 'sans-serif'
        if os.name == 'nt':
            plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'BIZ UDGothic', 'BIZ UDMincho']
        else:
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Hiragino Kaku Gothic Pro', 'Noto Sans CJK JP', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Takao']
        plt.rcParams['axes.unicode_minus'] = False
        return
    
    # フォント設定
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Windowsの場合
    if os.name == 'nt':
        plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'BIZ UDGothic', 'BIZ UDMincho']
    # Mac/Linuxの場合
    else:
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Hiragino Kaku Gothic Pro', 'Noto Sans CJK JP', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Takao']
    
    # ラベルなどのフォント設定
    plt.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示
    
    # フォント情報表示（まだ表示していない場合のみ）
    if verbose and not FONT_INFO_DISPLAYED:
        # フォント確認
        fonts = [f.name for f in fm.fontManager.ttflist]
        japanese_fonts = [f for f in fonts if any(keyword in f for keyword in ['Gothic', 'Mincho', 'Meiryo', 'Yu', 'ゴシック', '明朝', 'Hiragino', 'Noto'])]
        if japanese_fonts:
            print(f"利用可能な日本語フォント: {japanese_fonts[:5]}{'...' if len(japanese_fonts) > 5 else ''}")
        else:
            print("警告: 日本語フォントが見つかりませんでした。文字化けの可能性があります。")
        
        # 表示済みフラグを設定
        FONT_INFO_DISPLAYED = True
        # フラグファイルを作成
        with open(FONT_FLAG_FILE_PATH, 'w') as f:
            f.write("displayed")

def reset_font_info_flag():
    """
    フォント情報表示フラグをリセットする
    """
    global FONT_INFO_DISPLAYED
    FONT_INFO_DISPLAYED = False
    
    # フラグファイルがあれば削除
    if os.path.exists(FONT_FLAG_FILE_PATH):
        os.remove(FONT_FLAG_FILE_PATH)

def print_cuda_info(verbose=True, force=False):
    """
    CUDA情報を一度だけ表示するための関数
    
    Args:
        verbose (bool): 詳細な情報を表示するかどうか
        force (bool): 強制的に再表示するかどうか
    """
    global CUDA_INFO_DISPLAYED
    
    # フラグファイルがあれば表示済みとみなす
    if os.path.exists(FLAG_FILE_PATH) and not force:
        CUDA_INFO_DISPLAYED = True
    
    # すでに表示済みでかつ強制表示でなければスキップ
    if CUDA_INFO_DISPLAYED and not force:
        return
        
    # デバイスを確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 静かモードの場合は最小限の情報だけ
    if not verbose:
        if torch.cuda.is_available():
            print(f"GPU使用中 / GPU active: {torch.cuda.get_device_name(0)}")
        else:
            print("CPUモードで実行中 / Running in CPU mode")
        CUDA_INFO_DISPLAYED = True
        # フラグファイルを作成
        with open(FLAG_FILE_PATH, 'w') as f:
            f.write("displayed")
        return
        
    # 通常モードでの表示
    print(f"使用デバイス / Device: {device}")
    
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"CUDA バージョン / Version: {cuda_version}")
        
        if cuda_version.startswith('12.'):
            print("CUDA 12が検出されました。最適化された設定を使用します。/ CUDA 12 detected. Using optimized settings.")
            # CUDA 12特有の最適化設定
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # GPU情報（簡潔に）
            device_count = torch.cuda.device_count()
            print(f"利用可能なGPU / Available GPUs: {device_count}台")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        else:
            print(f"CUDA {cuda_version} を使用中 / Using CUDA {cuda_version}")
    else:
        print("GPUが利用できません。CPUで実行します。/ No GPU available. Running on CPU.")
    
    # 表示済みフラグを設定
    CUDA_INFO_DISPLAYED = True
    # フラグファイルを作成
    with open(FLAG_FILE_PATH, 'w') as f:
        f.write("displayed")

def reset_cuda_info_flag():
    """
    CUDA情報表示フラグをリセットする（デバッグ用）
    """
    global CUDA_INFO_DISPLAYED
    CUDA_INFO_DISPLAYED = False
    
    # フラグファイルがあれば削除
    if os.path.exists(FLAG_FILE_PATH):
        os.remove(FLAG_FILE_PATH)

def get_device():
    """現在のデバイスを取得"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 