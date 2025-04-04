from typing import Dict, List, Optional, Tuple, Any
import torch
import logging
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from src.training.experiment import Experiment
from src.training.callbacks_classifier import Callback, ReduceLROnPlateau, ModelCheckpoint

class ModelTrainer:
    """
    Class to manage model training with support for experiments and callback.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        experiment: Experiment,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            optimizer: The optimizer to use
            criterion: The loss function
            experiment: The experiment for tracking and logging
            device: The device on which to perform the training
            logger: Optional Logger for logging
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.experiment = experiment
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        self.model = self.model.to(self.device)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            dataloader (DataLoader): The DataLoader for the training data.

        Returns:
            Dict[str, float]: A dictionary containing the average loss and various metrics for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        predictions: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_metrics = self._calculate_metrics(targets, predictions)
        epoch_metrics['loss'] = epoch_loss
        
        return epoch_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Args:
            dataloader (DataLoader): The DataLoader for the validation data.

        Returns:
            Dict[str, float]: A dictionary containing the average loss and various metrics for the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        predictions: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(dataloader.dataset)
        val_metrics = self._calculate_metrics(targets, predictions)
        val_metrics['loss'] = val_loss
        
        return val_metrics
    
    def _calculate_metrics(self, targets: List[int], predictions: List[int]) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            targets: True Labels
            predictions: Predictions from model
            
        Returns:
            Dictionary containing the calculated metrics
        """
        return {
            'accuracy': np.mean(np.array(predictions) == np.array(targets)),
            'precision': precision_score(targets, predictions, average='weighted', zero_division=1),
            'recall': recall_score(targets, predictions, average='weighted', zero_division=1),
            'f1': f1_score(targets, predictions, average='weighted')
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        callbacks: Optional[List[Callback]] = None,
        resume_from: Optional[str] = None
    ) -> nn.Module:
        """
        Train the model for a specified number of epochs.

        Args:
            train_loader (DataLoader): The DataLoader for the training data.
            val_loader (DataLoader): The DataLoader for the validation data.
            num_epochs (int): The number of epochs to train for.
            callbacks (List[Any]): A list of callback objects for various training events.
            resume_from (str): If set, the checkpoint will load and resume training from where it left off.

        Returns:
            nn.Module: The trained model.
        """

        callbacks = callbacks or []
        
        if resume_from:
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.experiment.set_state(checkpoint['experiment_state'])
            start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Ripreso training dall'epoca {start_epoch}")
        else:
            start_epoch = 1
        
        self.logger.info(f"Start training for {num_epochs} epoche")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        self.logger.info(f"Criterion: {self.criterion.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{num_epochs}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            train_metrics = self.train_epoch(train_loader)
            
            val_metrics = self.validate(val_loader)
            val_metrics_prefixed = {f'val_{k}': v for k, v in val_metrics.items()}

            metrics = {**train_metrics, **val_metrics_prefixed}
            self.experiment.save_history('train', **train_metrics, lr=current_lr)
            self.experiment.save_history('val', **val_metrics)
            
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Epoca {epoch} - {metrics_str}")
            
            self.experiment.update_plots()

            stop_training = False
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    callback.on_epoch_end(epoch, metrics, self.model, self.optimizer, self.experiment)
                    self.logger.info(f"ModelCheckpoint: Saved model at epoch {epoch}")
                elif isinstance(callback, ReduceLROnPlateau):
                    old_lr = self.optimizer.param_groups[0]['lr']
                    callback.on_epoch_end(epoch, metrics)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if old_lr != new_lr:
                        self.logger.info(f"ReduceLROnPlateau: Learning rate changed from {old_lr} to {new_lr}")
                else:
                    stop_training = callback.on_epoch_end(epoch, metrics)
                    if stop_training:
                        self.logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

            if stop_training:
                break
            
            self.experiment.increment_epoch()
        
        self.logger.info("Training completated")
        return self.model
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from the model for the entire dataset.

        Args:
            dataloader (DataLoader): DataLoader containing the dataset to predict on.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array contains the true labels.
                - The second array contains the predicted labels.
        """
        self.model.eval()
        all_predictions: List[int] = []
        all_targets: List[int] = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.numpy())
        
        return np.array(all_targets), np.array(all_predictions)