"""
分散学習のサポート
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict, Any, Callable

class DistributedTrainer:
    """分散学習トレーナー"""
    
    def __init__(self,
                 experiment_class: type,
                 config: Dict[str, Any],
                 num_nodes: int = 1,
                 gpus_per_node: int = 1):
        self.experiment_class = experiment_class
        self.config = config
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.world_size = num_nodes * gpus_per_node
        
    def setup(self, rank: int, world_size: int):
        """分散環境のセットアップ"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # プロセスグループの初期化
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
    def cleanup(self):
        """分散環境のクリーンアップ"""
        dist.destroy_process_group()
        
    def train(self, rank: int, world_size: int):
        """1プロセスでの訓練"""
        # 分散環境のセットアップ
        self.setup(rank, world_size)
        
        # GPUの設定
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        # 実験インスタンスの作成
        experiment = self.experiment_class(self.config)
        
        # モデルをGPUに移動
        experiment.model = experiment.model.to(device)
        
        # DDPでモデルをラップ
        experiment.model = DDP(
            experiment.model,
            device_ids=[rank],
            output_device=rank
        )
        
        # データローダーの分散化
        train_sampler = DistributedSampler(
            experiment.train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        
        experiment.train_loader = torch.utils.data.DataLoader(
            experiment.train_dataset,
            batch_size=experiment.config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # 訓練の実行
        experiment.run(experiment.config.num_epochs)
        
        # クリーンアップ
        self.cleanup()
        
    def run_distributed(self):
        """分散訓練の実行"""
        mp.spawn(
            self.train,
            args=(self.world_size,),
            nprocs=self.world_size,
            join=True
        )
        
class DistributedExperimentMixin:
    """分散実験用のMixin"""
    
    def prepare_distributed(self, rank: int, world_size: int):
        """分散環境の準備"""
        # データセットの分散化
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # モデルの分散化
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank
        )
        
    def train_epoch_distributed(self) -> Dict[str, float]:
        """分散環境での1エポックの訓練"""
        self.model.train()
        total_loss = torch.zeros(1).to(self.device)
        correct = torch.zeros(1).to(self.device)
        total = torch.zeros(1).to(self.device)
        
        # サンプラーのエポック設定
        self.train_sampler.set_epoch(self.current_epoch)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 精度の計算
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()
            total += target.size(0)
            
        # 全プロセスでの集計
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        
        metrics = {
            'loss': total_loss.item() / len(self.train_loader),
            'accuracy': 100. * correct.item() / total.item()
        }
        
        return metrics
        
    def validate_distributed(self) -> Dict[str, float]:
        """分散環境での検証"""
        self.model.eval()
        total_loss = torch.zeros(1).to(self.device)
        correct = torch.zeros(1).to(self.device)
        total = torch.zeros(1).to(self.device)
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # 精度の計算
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum()
                total += target.size(0)
                
        # 全プロセスでの集計
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        
        metrics = {
            'loss': total_loss.item() / len(self.test_loader),
            'accuracy': 100. * correct.item() / total.item()
        }
        
        return metrics 