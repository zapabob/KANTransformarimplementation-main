"""
CUDAデバイス情報を管理するユーティリティ関数
"""

import torch
import psutil
import os
from typing import Dict, Optional, Tuple

def get_device() -> torch.device:
    """利用可能な最適なデバイスを取得する

    Returns:
        torch.device: CUDAが利用可能な場合はGPU、そうでない場合はCPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_cuda_info() -> Dict[str, str]:
    """CUDA環境の情報を取得する

    Returns:
        Dict[str, str]: CUDA情報を含む辞書
    """
    info = {
        'device': str(get_device()),
        'cuda_available': str(torch.cuda.is_available())
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': str(torch.cuda.device_count()),
            'current_device': str(torch.cuda.current_device())
        })
    
    return info

def print_cuda_info(verbose: bool = False) -> None:
    """CUDA環境の情報を表示する

    Args:
        verbose: 詳細情報を表示するかどうか
    """
    info = get_cuda_info()
    print("\n=== CUDA環境情報 ===")
    print(f"デバイス: {info['device']}")
    print(f"CUDA利用可能: {info['cuda_available']}")
    
    if torch.cuda.is_available():
        print(f"CUDAバージョン: {info['cuda_version']}")
        print(f"GPU名: {info['gpu_name']}")
        print(f"GPU数: {info['gpu_count']}")
        print(f"現在のデバイス: {info['current_device']}")
        
        if verbose:
            memory = get_memory_info()
            print("\n=== メモリ情報 ===")
            print(f"CUDA割り当て済み: {memory['cuda_allocated']:.2f} GB")
            print(f"CUDAキャッシュ: {memory['cuda_cached']:.2f} GB")
            print("\n=== システムメモリ ===")
            print(f"合計: {memory['system_total']:.2f} GB")
            print(f"利用可能: {memory['system_available']:.2f} GB")
            print(f"使用中: {memory['system_used']:.2f} GB")
            
            capability = get_cuda_capability()
            if capability:
                print(f"\nCUDA Compute Capability: {capability[0]}.{capability[1]}")

def get_memory_info() -> Dict[str, float]:
    """メモリ使用状況を取得する

    Returns:
        Dict[str, float]: メモリ情報を含む辞書（単位: GB）
    """
    info = {
        'system_total': psutil.virtual_memory().total / (1024**3),
        'system_available': psutil.virtual_memory().available / (1024**3),
        'system_used': psutil.virtual_memory().used / (1024**3)
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_allocated': torch.cuda.memory_allocated(0) / (1024**3),
            'cuda_cached': torch.cuda.memory_reserved(0) / (1024**3)
        })
    
    return info

def set_cuda_device(device_id: Optional[int] = None) -> None:
    """使用するCUDAデバイスを設定する

    Args:
        device_id: 使用するデバイスのID（指定がない場合は0）
    """
    if torch.cuda.is_available():
        if device_id is None:
            device_id = 0
        torch.cuda.set_device(device_id)

def get_optimal_worker_info() -> Tuple[int, int]:
    """データローダーの最適なワーカー数とバッチサイズを取得する

    Returns:
        Tuple[int, int]: (ワーカー数, バッチサイズ)
    """
    cpu_count = os.cpu_count() or 1
    workers = min(cpu_count - 1, 8)  # 1コアは他の処理用に残す
    
    # GPU利用可能な場合は大きめのバッチサイズ
    if torch.cuda.is_available():
        batch_size = 128
    else:
        batch_size = 32
    
    return workers, batch_size

def clear_cuda_cache() -> None:
    """CUDAキャッシュをクリアする"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_cuda_capability() -> Optional[Tuple[int, int]]:
    """現在のGPUのCUDA Compute Capabilityを取得する

    Returns:
        Optional[Tuple[int, int]]: (メジャーバージョン, マイナーバージョン)
        GPUが利用できない場合はNone
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return major, minor
    return None 