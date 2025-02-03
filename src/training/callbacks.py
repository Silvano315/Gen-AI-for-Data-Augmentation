import torch
import matplotlib.pyplot as plt
from abc import ABC
from typing import Dict, Any, Optional, Union
from pathlib import Path

__all__ = ["Callback", "EarlyStopping", "ModelCheckpoint", "MetricsHistory", "LearningRateScheduler"]

class Callback(ABC):
    """Base class for all callbacks."""
    
    def on_train_begin(self, trainer: Any):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: Any):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer: Any, batch: Any):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer: Any, batch: Any, logs: Dict[str, float]):
        """Called at the end of each batch."""
        pass
    
    def on_evaluate_begin(self, trainer: Any):
        """Called at the beginning of evaluation."""
        pass
    
    def on_evaluate_end(self, trainer: Any, metrics: Dict[str, float]):
        """Called at the end of evaluation."""
        pass

class EarlyStopping(Callback):
    """Early stopping callback based on monitoring metric."""
    
    def __init__(
        self,
        monitor: str = 'fid',
        min_delta: float = 0.0,
        patience: int = 5,
        mode: str = 'min',
        baseline: Optional[float] = None
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.baseline = baseline
        
        self.wait = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.stopped_epoch = 0

    def on_train_begin(self, trainer: Any):
        self.wait = 0
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        
    def on_evaluate_end(self, trainer: Any, metrics: Dict[str, float]):
        current = metrics.get(self.monitor)
        if current is None:
            return
        
        if self.baseline is not None:
            if self.mode == 'min' and current < self.baseline:
                trainer.stop_training = True
            elif self.mode == 'max' and current > self.baseline:
                trainer.stop_training = True
            return
            
        if self.mode == 'min':
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
        else:
            if current > self.best + self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                
        if self.wait >= self.patience:
            self.stopped_epoch = trainer.current_epoch
            trainer.stop_training = True
            trainer.logger.info(
                f'Early stopping triggered at epoch {self.stopped_epoch}'
            )

class ModelCheckpoint(Callback):
    """Save model checkpoints based on monitoring metric."""
    
    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = 'fid',
        save_best_only: bool = True,
        save_freq: Optional[int] = None,
        mode: str = 'min'
    ):
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.mode = mode
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        
    def _save_checkpoint(self, trainer: Any, filepath: Path):
        """Save model checkpoint."""
        checkpoint = {
            'generator_state': trainer.gan.generator.state_dict(),
            'discriminator_state': trainer.gan.discriminator.state_dict(),
            'g_optimizer': trainer.gan.g_optimizer.state_dict(),
            'd_optimizer': trainer.gan.d_optimizer.state_dict(),
            'g_scheduler': trainer.gan.g_scheduler.state_dict(),
            'd_scheduler': trainer.gan.d_scheduler.state_dict(),
            'epoch': trainer.current_epoch,
            'step': trainer.current_step,
            'best_metrics': trainer.logger.best_metrics
        }
        
        torch.save(checkpoint, filepath)
        trainer.logger.info(f'Model checkpoint saved to {filepath}')
        
    def on_evaluate_end(self, trainer: Any, metrics: Dict[str, float]):
        current = metrics.get(self.monitor)
        if current is None:
            return
            
        if self.save_best_only:
            if self.mode == 'min' and current < self.best:
                self.best = current
                self._save_checkpoint(trainer, self.filepath)
            elif self.mode == 'max' and current > self.best:
                self.best = current
                self._save_checkpoint(trainer, self.filepath)
                
    def on_epoch_end(self, trainer: Any, epoch: int):
        if self.save_freq and epoch % self.save_freq == 0:
            filepath = self.filepath.parent / f'checkpoint_epoch_{epoch}.pt'
            self._save_checkpoint(trainer, filepath)

class MetricsHistory(Callback):
    """Track and plot training metrics history."""
    
    def __init__(self, log_dir: Union[str, Path]):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, list] = {}
        
    def on_batch_end(self, trainer: Any, batch: Any, logs: Dict[str, float]):
        for metric, value in logs.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(value)
            
    def on_train_end(self, trainer: Any):
        try:
            for metric, values in self.history.items():
                plt.figure(figsize=(10, 6))
                plt.plot(values)
                plt.title(f'{metric} History')
                plt.xlabel('Batch')
                plt.ylabel(metric)
                plt.savefig(self.log_dir / f'{metric}_history.png')
                plt.close()
                
            trainer.logger.info(
                f'Metrics plots saved to {self.log_dir}'
            )
        except ImportError:
            trainer.logger.warning(
                'matplotlib not installed, skipping metrics plots'
            )

class LearningRateScheduler(Callback):
    """Callback for dynamic learning rate scheduling."""
    
    def __init__(
        self,
        monitor: str = 'fid',
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 1e-6,
        mode: str = 'min'
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        
        self.wait = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        
    def _get_lr(self, optimizer: torch.optim.Optimizer) -> float:
        """Get current learning rate."""
        return optimizer.param_groups[0]['lr']
        
    def _set_lr(self, optimizer: torch.optim.Optimizer, lr: float):
        """Set new learning rate."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def on_evaluate_end(self, trainer: Any, metrics: Dict[str, float]):
        current = metrics.get(self.monitor)
        if current is None:
            return
            
        if self.mode == 'min':
            if current < self.best:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
        else:
            if current > self.best:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                
        if self.wait >= self.patience:
            for opt_name in ['g_optimizer', 'd_optimizer']:
                optimizer = getattr(trainer.gan, opt_name)
                old_lr = self._get_lr(optimizer)
                new_lr = max(old_lr * self.factor, self.min_lr)
                self._set_lr(optimizer, new_lr)
                
                trainer.logger.info(
                    f'Reduced {opt_name} learning rate: {old_lr:.2e} -> {new_lr:.2e}'
                )
            
            self.wait = 0