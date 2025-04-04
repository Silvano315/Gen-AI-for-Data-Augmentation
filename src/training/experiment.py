import numpy as np
from pathlib import Path
import os
import torch
import shutil
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class Experiment:
    """
    A class to manage machine learning experiments, including logging, 
    saving/loading weights, and visualizing training history.
    """
    def __init__(self, name: str, root: str, logger: Optional[logging.Logger] = None) -> None:
        """
        Initializes a new experiment.
        
        Args:
            name: Name of experiment
            root: Root directory for the experiment files
            logger: Optional logger for logging
        """
        self.name = name
        self.root = Path(root) / name
        self.logger = logger or self._setup_default_logger()
        self.epoch = 1
        #self.best_val_loss = sys.float_info.max
        self.best_val_loss = float('inf')
        self.best_val_loss_epoch = 1
        self.weights_dir = os.path.join(self.root, 'weights')
        self.history_dir = os.path.join(self.root, 'history')
        self.results_dir = os.path.join(self.root, 'results')
        self.latest_weights = os.path.join(self.weights_dir, 'latest_weights.pth')
        self.latest_optimizer = os.path.join(self.weights_dir, 'latest_optim.pth')
        self.best_weights_path = self.latest_weights
        self.best_optimizer_path = self.latest_optimizer
        self.train_history_fpath = os.path.join(self.history_dir, 'train.csv')
        self.val_history_fpath = os.path.join(self.history_dir, 'val.csv')
        self.test_history_fpath = os.path.join(self.history_dir, 'test.csv')
        self.metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'lr']
        self.history = {split: {metric: [] for metric in self.metrics} for split in ['train', 'val', 'test']}

    def _setup_default_logger(self) -> logging.Logger:
        """Set a default logger if one is not provided."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger

    def log(self, msg: str) -> None:
        """Log a message using the configured logger.
        
        Args:
            msg: message to log
        """
        if self.logger:
            self.logger.info(msg)

    def init(self) -> None:
        """Initializes a new experiment by creating the necessary directories."""
        self.log("Creating new experiment")
        self.init_dirs()
        self.init_history_files()

    def init_dirs(self):
        """Create the directories needed for the experiment."""
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def init_history_files(self):
        """Initialize the CSV files to track training history."""
        header = ','.join(['epoch'] + self.metrics) + '\n'
        for split in ['train', 'val', 'test']:
            fpath = getattr(self, f'{split}_history_fpath')
            with open(fpath, 'w') as f:
                f.write(header)

    def increment_epoch(self) -> None:
        """Increase the counter of epochs."""
        self.epoch += 1

    def resume(self, model: torch.nn.Module, optim: torch.optim.Optimizer, weights_fpath: str = None, optim_path: str = None):
        self.log("Resuming existing experiment")
        if weights_fpath is None:
            weights_fpath = self.latest_weights
        if optim_path is None:
            optim_path = self.latest_optimizer

        model, state = self.load_weights(model, weights_fpath)
        optim = self.load_optimizer(optim, optim_path)

        self.best_val_loss = state['best_val_loss']
        self.best_val_loss_epoch = state['best_val_loss_epoch']
        self.epoch = state['last_epoch'] + 1
        self.load_history_from_file('train')
        self.load_history_from_file('val')

        return model, optim

    def load_history_from_file(self, split: str):
        """
        Loads training history metrics from a CSV file for a specific data split.
        
        Args:
            split (str): The data split to load history for (e.g. 'train', 'val')
        """
        fpath = getattr(self, f'{split}_history_fpath')
        data = np.loadtxt(fpath, delimiter=',', skiprows=1)
        if data.ndim == 1:  
            data = data.reshape(1, -1) 
        for i, metric in enumerate(self.metrics):
            self.history[split][metric] = data[:, i+1].tolist()

    def save_history(self, split: str, **kwargs):
        """
        Saves training history metrics.
        
        Args:
            split: The split of data ('train', 'val', 'test')
            **kwargs: Metrics to be saved
        """
        for metric, value in kwargs.items():
            if metric == 'lr':
                self.history['train']['lr'].append(value) 
            else:
                metric_name = metric[4:] if metric.startswith('val_') else metric
                if metric_name not in self.history[split]:
                    self.history[split][metric_name] = []
                self.history[split][metric_name].append(value)
        
        fpath = getattr(self, f'{split}_history_fpath')
        with open(fpath, 'a') as f:
            values = [str(kwargs.get(metric, kwargs.get(f'val_{metric}', ''))) for metric in self.metrics]
            f.write(f"{self.epoch},{','.join(values)}\n")
        
        if split == 'val' and 'loss' in kwargs:
            if self.is_best_loss(kwargs['loss']):
                self.best_val_loss = kwargs['loss']
                self.best_val_loss_epoch = self.epoch

    def is_best_loss(self, loss: float) -> bool:
        """Check if the current loss is the best so far."""
        return loss < self.best_val_loss

    def save_weights(self, model: torch.nn.Module, **kwargs):
        """
        Save the weights of the model.
        
        Args:
            model: Model to save
            **kwargs: Additional metrics to save with weights
        """
        weights_fname = f"{self.name}-weights-{self.epoch}-" + "-".join([f"{v:.3f}" for v in kwargs.values()]) + ".pth"
        weights_fpath = os.path.join(self.weights_dir, weights_fname)
        try:
            torch.save({
                'last_epoch': self.epoch,
                'best_val_loss': self.best_val_loss,
                'best_val_loss_epoch': self.best_val_loss_epoch,
                'experiment': self.name,
                'state_dict': model.state_dict(),
                **kwargs
            }, weights_fpath)
            shutil.copyfile(weights_fpath, self.latest_weights)
            if 'val_loss' in kwargs and self.is_best_loss(kwargs['val_loss']):
                self.best_weights_path = weights_fpath
            self.log(f"Successfully saved weights to {weights_fpath}")
        except Exception as e:
            self.log(f"Error saving weights: {str(e)}")
            raise

    def load_weights(self, model: torch.nn.Module, fpath: str):
        """
        Load the weights of the model.
        
        Args:
            model: The model in which to load weights
            fpath: Optional path of the weights to be loaded (use latest_weights if None)
            
        Returns:
            Tuple containing the uploaded model and state dictionary
        """
        self.log(f"Loading weights from '{fpath}'")
        try:
            state = torch.load(fpath)
            model.load_state_dict(state['state_dict'])
            self.log(f"Loaded weights from experiment {self.name} (last_epoch {state['last_epoch']})")
            return model, state
        except FileNotFoundError:
            self.log(f"Error: Weights file not found at {fpath}")
            raise
        except RuntimeError as e:
            self.log(f"Error loading state dict: {str(e)}")
            raise

    def save_optimizer(self, optimizer: torch.optim.Optimizer, val_loss: float):
        """
        Saves the optimizer state to disk and handles best model tracking based on validation loss.
        
        Args:
            optimizer (torch.optim.Optimizer): The PyTorch optimizer to save
            val_loss (float): The current validation loss used to track best model
        """
        optim_fname = f"{self.name}-optim-{self.epoch}.pth"
        optim_fpath = os.path.join(self.weights_dir, optim_fname)
        try:
            torch.save({
                'last_epoch': self.epoch,
                'experiment': self.name,
                'state_dict': optimizer.state_dict()
            }, optim_fpath)
            shutil.copyfile(optim_fpath, self.latest_optimizer)
            if self.is_best_loss(val_loss):
                self.best_optimizer_path = optim_fpath
            self.log(f"Successfully saved optimizer to {optim_fpath}")
        except Exception as e:
            self.log(f"Error saving optimizer: {str(e)}")
            raise

    def load_optimizer(self, optimizer: torch.optim.Optimizer, fpath: str):
        """
        Loads an optimizer state from a checkpoint file.
        
        Args:
            optimizer (torch.optim.Optimizer): The optimizer to load the state into
            fpath (str): Path to the checkpoint file
            
        Returns:
            torch.optim.Optimizer: The optimizer with loaded state
        """
        self.log(f"Loading optimizer from '{fpath}'")
        try:
            optim = torch.load(fpath)
            optimizer.load_state_dict(optim['state_dict'])
            self.log(f"Successfully loaded optimizer from session {optim['experiment']}, last_epoch {optim['last_epoch']}")
            return optimizer
        except FileNotFoundError:
            self.log(f"Error: Optimizer file not found at {fpath}")
            raise
        except Exception as e:
            self.log(f"Error loading optimizer: {str(e)}")
            raise

    def save_checkpoint(self, model, optimizer, epoch, logs):
        """
        Creates a complete training checkpoint including model weights and optimizer state.
        
        Args:
            model: The PyTorch model to save
            optimizer (torch.optim.Optimizer): The optimizer to save
            epoch (int): Current training epoch
            logs (dict): Dictionary containing training metrics including optional 'val_loss'
        """
        self.save_weights(model, **logs)
        if 'val_loss' in logs:
            self.save_optimizer(optimizer, logs['val_loss'])
        else:
            self.save_optimizer(optimizer, float('inf'))

    def load_checkpoint(self, model, optimizer):
        """
        Restores a complete training checkpoint, loading both model weights and optimizer state.
        
        Args:
            model: The PyTorch model to load weights into
            optimizer (torch.optim.Optimizer): The optimizer to restore state into
            
        Returns:
            tuple: (model, optimizer) The model and optimizer with restored states
        """
        model = self.load_weights(model)
        optimizer = self.load_optimizer(optimizer)
        return model, optimizer
    
    def cleanup_old_files(self, keep_last_n: int = 1):
        def get_sorted_files(prefix):
            files = [f for f in os.listdir(self.weights_dir) if f.startswith(prefix)]
            return sorted(files, key=lambda x: os.path.getmtime(os.path.join(self.weights_dir, x)), reverse=True)

        for prefix in [f"{self.name}-weights-", f"{self.name}-optim-"]:
            files = get_sorted_files(prefix)
            files_to_keep = set(files[:keep_last_n])
            files_to_keep.add(os.path.basename(self.latest_weights))
            files_to_keep.add(os.path.basename(self.latest_optimizer))
            files_to_keep.add(os.path.basename(self.best_weights_path))
            files_to_keep.add(os.path.basename(self.best_optimizer_path))

            for file in files:
                if file not in files_to_keep:
                    os.remove(os.path.join(self.weights_dir, file))
                    self.log(f"Removed old file: {file}")

    def get_state(self):
        """Get the current state of the experiment."""
        return {
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_loss_epoch': self.best_val_loss_epoch,
            'history': self.history
        }

    def set_state(self, state):
        """
        Sets the state of the experiment.
        
        Args:
            state: Dictionary containing the state of the experiment
        """
        self.epoch = state['epoch']
        self.best_val_loss = state['best_val_loss']
        self.best_val_loss_epoch = state['best_val_loss_epoch']
        self.history = state['history']

    def plot_history(self):
        """Generate the plots of the training history."""
        for metric in self.metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            for split in ['train', 'val']:
                ax.plot(self.history[split][metric], label=split.capitalize())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.set_title(f'{self.name} - {metric.capitalize()}')
            plt.savefig(os.path.join(self.history_dir, f'{metric}.png'))
            plt.close()

        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(12, 6*len(self.metrics)))
        for i, metric in enumerate(self.metrics):
            for split in ['train', 'val']:
                axes[i].plot(self.history[split][metric], label=split.capitalize())
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].set_title(f'{metric.capitalize()}')
        fig.suptitle(f'{self.name} - Training History')
        plt.tight_layout()
        plt.savefig(os.path.join(self.history_dir, 'combined_history.png'))
        plt.close()

        if 'lr' in self.history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.history['lr'], label='Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')  
            ax.legend()
            ax.set_title(f'{self.name} - Learning Rate')
            plt.savefig(os.path.join(self.history_dir, 'learning_rate.png'))
            plt.close()

    def update_plots(self):
        """Update history plots after epoch."""
        self.plot_history()

    def calculate_average_metrics(self, split: str, last_n_epochs: int = 5) -> Dict[str, float]:
        """
        Calculate average metrics for the last n epochs.

        Args:
            split (str): The data split to calculate metrics for ('train', 'val', or 'test').
            last_n_epochs (int): Number of last epochs to consider for averaging.

        Returns:
            Dict[str, float]: A dictionary of averaged metrics.
        """
        avg_metrics = {}
        for metric in self.metrics:
            values = self.history[split][metric][-last_n_epochs:]
            avg_metrics[metric] = sum(values) / len(values)
        return avg_metrics

    def export_results_to_json(self, filepath: str):
        """
        Export experiment results to a JSON file.

        Args:
            filepath (str): Path to save the JSON file.
        """
        results = {
            "name": self.name,
            "best_val_loss": self.best_val_loss,
            "best_val_loss_epoch": self.best_val_loss_epoch,
            "final_metrics": {
                split: self.calculate_average_metrics(split) 
                for split in ['train', 'val', 'test']
            },
            "history": self.history
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
            self.log(f"Successfully exported results to {filepath}")
        except Exception as e:
            self.log(f"Error exporting results to JSON: {str(e)}")
            raise

    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """
        Get the epoch with the best performance for a given metric.

        Args:
            metric (str): The metric to consider.
            mode (str): 'min' if lower is better, 'max' if higher is better.

        Returns:
            int: The epoch with the best performance.
        """
        values = self.history['val'][metric]
        if mode == 'min':
            best_value = min(values)
        elif mode == 'max':
            best_value = max(values)
        else:
            raise ValueError("Mode must be 'min' or 'max'")
        return values.index(best_value) + 1  

    def plot_learning_rate(self, lr_history: List[float]):
        """
        Plot the learning rate over epochs.

        Args:
            lr_history (List[float]): List of learning rates for each epoch.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(lr_history) + 1), lr_history)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'{self.name} - Learning Rate Schedule')
        plt.savefig(os.path.join(self.history_dir, 'learning_rate.png'))
        plt.close()