import os
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

def calculate_mean_std(data):
    return np.mean(data), np.std(data)

def scatter_plot_metrics(train_csv_path: str, val_csv_path: str) -> None:
    """
    Plot chosen metrics from training and validation CSV files using Plotly.
    
    Args:
        train_csv_path (str): Path to the CSV file containing training metrics.
        val_csv_path (str): Path to the CSV file containing validation metrics.
    
    Returns:
        None: Displays an interactive scatter plot of the training and validation metrics.
    """
    
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    epochs = train_df['epoch'].tolist()
    
    fig = go.Figure()
    dropdown_buttons = []
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    
    for i, metric in enumerate(metrics):
        train_mean, train_std = calculate_mean_std(train_df[metric])
        val_mean, val_std = calculate_mean_std(val_df[metric])
        
        fig.add_trace(go.Scatter(
            x=epochs, y=train_df[metric], mode='lines+markers', visible=(i == 0),
            name=f'Train {metric.capitalize()} (Mean: {train_mean:.2f}, Std: {train_std:.2f})'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=val_df[metric], mode='lines+markers', visible=(i == 0),
            name=f'Val {metric.capitalize()} (Mean: {val_mean:.2f}, Std: {val_std:.2f})'
        ))
        
        dropdown_buttons.append({
            'label': metric.capitalize(),
            'method': 'update',
            'args': [
                {'visible': [False] * len(metrics) * 2},
                {'title': f'{metric.capitalize()}'}
            ]
        })
        dropdown_buttons[-1]['args'][0]['visible'][i * 2] = True
        dropdown_buttons[-1]['args'][0]['visible'][i * 2 + 1] = True
    
    fig.update_layout(
        title='Model Performance per Epoch',
        xaxis_title='Epoch',
        yaxis_title='Metric Value',
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
        }],
        legend_title="Legend"
    )
    
    fig.show()



def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], path_to_save: str) -> None:
    """
    Plot and save a confusion matrix.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        classes (List[str]): List of class names.
        path_to_save (str): Path where to save confusion matrix png

    Returns:
        None: This function saves the plot as a file and doesn't return anything.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(path_to_save+"images", exist_ok=True)
    plt.savefig(path_to_save+'images/confusion_matrix.png')
    plt.close()

def plot_misclassified_images(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_images: int = 9,
    class_names: Optional[List[str]] = None,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> None:
    """
    Displays misclassified images from the model in an nxn subplot.

    Args:
        model (torch.nn.Module): The trained model
        dataloader (DataLoader): DataLoader containing the dataset
        device (torch.device): Device to run the model on (CPU or GPU)
        num_images (int): Number of images to display (default: 9)
        class_names (Optional[List[str]]): List of class names (default: None)
        mean (Tuple[float, float, float]): Mean used for normalization (default: (0.5, 0.5, 0.5))
        std (Tuple[float, float, float]): Standard deviation used for normalization (default: (0.5, 0.5, 0.5))
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            incorrect = preds != labels
            if incorrect.any():
                misclassified_images.extend(images[incorrect].cpu())
                misclassified_labels.extend(labels[incorrect].cpu())
                misclassified_preds.extend(preds[incorrect].cpu())
            
            if len(misclassified_images) >= num_images:
                break
    
    n = int(math.sqrt(num_images))
    if n * n < num_images:
        n += 1
    
    fig = plt.figure(figsize=(15, 15))
    for idx in range(min(num_images, len(misclassified_images))):
        ax = fig.add_subplot(n, n, idx + 1)
        
        img = misclassified_images[idx].permute(1, 2, 0)
        
        mean_tensor = torch.tensor(mean)
        std_tensor = torch.tensor(std)
        img = img * std_tensor + mean_tensor
        img = torch.clamp(img, 0, 1)
        
        ax.imshow(img)
        
        true_label = misclassified_labels[idx].item()
        pred_label = misclassified_preds[idx].item()
        
        if class_names:
            title = f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}'
        else:
            title = f'True: {true_label}\nPred: {pred_label}'
            
        ax.set_title(title, color='red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()