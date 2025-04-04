from typing import Dict, Optional
import numpy as np
import torch
from src.training.experiment import Experiment

class Callback:
    """Basic class for training callbacks."""
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> bool:
        """
        Called at the end of every epoch.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary containing the metrics of the era
            
        Returns:
            bool: True if training is to be stopped
        """
        return False
    

class EarlyStopping(Callback):
    """
    Callback to stop training when a metric stops improving.
    """
    
    def __init__(
        self, 
        monitor: str = 'val_loss',
        min_delta: float = 0,
        patience: int = 0,
        verbose: bool = False,
        mode: str = 'auto'
    ) -> None:
        """
        Args:
            monitor:Metric to be monitored
            min_delta: Minimum change to be considered as improvement
            patience: Number of times to wait before stopping
            verbose: Whether to print messages
            mode: 'min', 'max' o 'auto'
        """
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.best: Optional[float] = None
        self.mode = mode
        self.monitor_op = None
        self._init_monitor_op()

    def _init_monitor_op(self):
        """Initializes the mode-based comparison operator."""
        if self.mode not in ['auto', 'min', 'max']:
            print(f'EarlyStopping mode {self.mode} is unknown, fallback to auto mode.')
            self.mode = 'auto'
        
        if self.mode == 'min' or (self.mode == 'auto' and 'loss' in self.monitor):
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> bool:
        """
        Determines whether to stop training early based on a monitored metric at the end of an epoch.

        Args:
            epoch : int
                The index of the current epoch.
            logs : Dict[str, float]
                A dictionary containing the metrics logged for the current epoch. The monitored metric
                must be present in this dictionary.

        Returns:
            bool
                True if early stopping is triggered, otherwise False.
        """
        current = logs.get(self.monitor)
        if current is None:
            print(f"Early stopping conditioned on metric `{self.monitor}` which is not available. "
                  f"Available metrics are: {','.join(list(logs.keys()))}")
            return False

        if self.best is None:
            self.best = current
            self.wait = 0
        elif self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f'Epoch {epoch}: early stopping')
                return True
        return False
    

class ModelCheckpoint(Callback):
    """
    Callback to save the model during training.
    """
    def __init__(self, filepath: str, monitor: str = 'val_loss', verbose: int = 0,
                 save_best_only: bool = False, mode: str = 'auto', keep_last_n: int = 1):
        """
        Args:
            filepath: Path where to save models
            monitor: Metric to be monitored
            verbose: Verbose level
            save_best_only: If you save only the best model
            mode: 'min', 'max' o 'auto'
        """
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = None
        self.monitor_op = None
        self.keep_last_n = keep_last_n
        self._init_monitor_op()

    def _init_monitor_op(self):
        """Initializes the mode-based comparison operator."""
        if self.mode not in ['auto', 'min', 'max']:
            print(f'ModelCheckpoint mode {self.mode} is unknown, fallback to auto mode.')
            self.mode = 'auto'

        if self.mode == 'min' or (self.mode == 'auto' and 'loss' in self.monitor):
            self.monitor_op = np.less
            self.best = float('inf')
        else:
            self.monitor_op = np.greater
            self.best = -float('inf')

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: torch.nn.Module, 
                     optimizer: torch.optim.Optimizer, experiment: Experiment):
        """
        Save the model's checkpoint at the end of the epoch if necessary.
        """
        current = logs.get(self.monitor)
        if current is None:
            print(f"Can't save best model, metric `{self.monitor}` is not available. "
                  f"Available metrics are: {','.join(list(logs.keys()))}")
            return

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f'\nEpoch {epoch:05d}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, '
                          f'saving model to {self.filepath}')
                self.best = current
                self._save_checkpoint(model, optimizer, epoch, logs, experiment)
        else:
            if self.verbose > 0:
                print(f'\nEpoch {epoch:05d}: saving model to {self.filepath}')
            self._save_checkpoint(model, optimizer, epoch, logs, experiment)
        
        experiment.cleanup_old_files(self.keep_last_n)

    def _save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                         epoch: int, logs: Dict[str, float], experiment: Experiment):
        """
        Saves a checkpoint of the current training state, including the model, optimizer,
        and relevant metadata.
        """
        experiment.save_checkpoint(model, optimizer, epoch, logs)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'logs': logs,
            'best': self.best,
            'experiment_state': experiment.get_state() 
        }
        
        torch.save(checkpoint, self.filepath)


class ReduceLROnPlateau(Callback):
    """
    Callback to reduce the learning rate when a monitored metric stops improving.

    Args:
    optimizer : The optimizer whose learning rate will be reduced.
    mode : One of {'min', 'max'}.
    factor : Factor by which the learning rate will be reduced. 
    patience : Number of epochs with no improvement after which learning rate will be reduced.
    verbose : If True, prints a message for each learning rate reduction. 
    min_lr : Minimum learning rate after reduction.
    eps : Minimum change in learning rate to consider a reduction. 
    monitor : The name of the metric to monitor.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, mode: str = 'min', factor: float = 0.1, 
                 patience: int = 10, verbose: bool = False, min_lr: float = 0, eps: float = 1e-8,
                 monitor: str = 'val_loss'):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.eps = eps
        self.monitor = monitor
        self.cooldown_counter = 0
        self.wait = 0
        self.best = None
        self.mode_worse = None
        self.is_better = None
        self._init_is_better(mode)

    def _init_is_better(self, mode):
        """
        Initializes the comparison logic based on the specified mode.

        Args:
        mode : One of {'min', 'max'}.
        """
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
            self.is_better = lambda a, best: a < best - self.eps
        if mode == 'max':
            self.mode_worse = -float('inf')
            self.is_better = lambda a, best: a > best + self.eps

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """
        Checks the monitored metric at the end of an epoch and updates the learning rate if necessary.

        Args:
        epoch : The index of the current epoch.
        logs : A dictionary containing the metrics logged for the current epoch. 
        """
        current = logs.get(self.monitor)
        if current is None:
            print(f"ReduceLROnPlateau conditioned on metric `{self.monitor}` which is not available. "
                  f"Available metrics are: {','.join(list(logs.keys()))}")
            return

        if self.best is None or self.is_better(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self._reduce_lr(epoch)
            self.wait = 0

    def _reduce_lr(self, epoch):
        """
        Reduces the learning rate for all parameter groups in the optimizer.

        Args:
        epoch : The index of the current epoch.
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}.')