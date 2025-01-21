from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class PetDatasetHandler:
    """Handles the loading and analysis of the Oxford-IIIT Pet Dataset.
    
    This class manages dataset loading and exploratory analysis
    for the pet image classification task.
    
    Attributes:
        data_dir (Path): Directory containing the dataset
        train_dataset (Dataset): Training dataset
        test_dataset (Dataset): Test dataset
    """
    
    def __init__(self, data_dir: Path) -> None:
        """Initialize the dataset handler.
        
        Args:
            data_dir (Path): Directory where the dataset will be stored/is stored
        """
        self.data_dir = data_dir
        self.train_dataset = None
        self.test_dataset = None
    
    def load_dataset(self, transform: Optional[callable] = None) -> Tuple[Dataset, Dataset]:
        """Load and split the dataset into train and test sets.
        
        Args:
            transform (callable, optional): Transformations to apply to images.
                                         If None, images will be loaded as-is.
        
        Returns:
            Tuple[Dataset, Dataset]: Training and test datasets
        """
        self.train_dataset = OxfordIIITPet(
            root=self.data_dir,
            split='trainval',
            transform=transform,
            download=True
        )
        
        self.test_dataset = OxfordIIITPet(
            root=self.data_dir,
            split='test',
            transform=transform,
            download=True
        )
        
        return self.train_dataset, self.test_dataset
    
    def get_dataset_info(self) -> Dict[str, int]:
        """Get basic information about the datasets.
        
        Returns:
            Dict[str, int]: Dictionary containing:
                - total_samples: Total number of samples
                - train_samples: Number of training samples
                - test_samples: Number of test samples
                - num_classes: Number of unique classes
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        info = {
            'total_samples': len(self.train_dataset) + len(self.test_dataset),
            'train_samples': len(self.train_dataset),
            'test_samples': len(self.test_dataset),
            'num_classes': len(self.train_dataset.classes)
        }
        return info
    
    def get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of classes in both training and test datasets.
        
        Returns:
            Dict[str, Dict[str, int]]: Dictionary containing class distributions for:
                - train: class distribution in training set
                - test: class distribution in test set
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        distributions = {'train': {}, 'test': {}}
        
        for _, class_idx in self.train_dataset:
            class_name = self.train_dataset.classes[class_idx]
            distributions['train'][class_name] = distributions['train'].get(class_name, 0) + 1
            
        for _, class_idx in self.test_dataset:
            class_name = self.test_dataset.classes[class_idx]
            distributions['test'][class_name] = distributions['test'].get(class_name, 0) + 1
            
        return distributions
    
    def plot_class_distribution(self) -> go.Figure:
        """Plot class distribution using Plotly.
        
        Returns:
            go.Figure: Plotly figure object showing train and test distributions
        """
        distributions = self.get_class_distribution()
        
        df_list = []
        for split, dist in distributions.items():
            temp_df = pd.DataFrame(list(dist.items()), columns=['Class', 'Count'])
            temp_df['Split'] = split
            df_list.append(temp_df)
        
        df = pd.concat(df_list, ignore_index=True)
        
        fig = px.bar(df, x='Class', y='Count', color='Split',
                    barmode='group',
                    title='Class Distribution in Training and Test Sets',
                    template='plotly_white')
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600
        )
        
        return fig
    
    def visualize_samples(self, num_samples: int = 9) -> go.Figure:
        """Visualize random samples from the training dataset.
        
        Args:
            num_samples (int): Number of samples to visualize (perfect square)
        
        Returns:
            go.Figure: Plotly figure with grid of images
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        n = int(np.sqrt(num_samples))
        fig = plt.figure(figsize=(15, 15))
        
        original_dataset = OxfordIIITPet(
            root=self.data_dir,
            split='trainval',
            transform=transforms.ToTensor(),  
            download=False
        )
        
        for i in range(num_samples):
            idx = np.random.randint(len(original_dataset))
            img, label = original_dataset[idx]
            
            img = img.permute(1, 2, 0).numpy()
            
            plt.subplot(n, n, i + 1)
            plt.imshow(img)
            plt.title(original_dataset.classes[label])
            plt.axis('off')
            
        plt.tight_layout()
        return fig
    
    def get_image_stats(self, sample_size: int = 100) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics about the images.
        
        Args:
            sample_size (int): Number of images to sample for statistics
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing image statistics:
                - dimensions: width and height statistics
                - aspect_ratio: aspect ratio statistics
                - file_size: file size statistics (in MB)
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        widths = []
        heights = []
        sizes = []
        
        indices = np.random.choice(len(self.train_dataset), 
                                 min(sample_size, len(self.train_dataset)))
        
        for idx in indices:
            img_path = self.train_dataset.images[idx]
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                sizes.append(Path(img_path).stat().st_size / (1024 * 1024))  
                
        stats = {
            'dimensions': {
                'mean_width': np.mean(widths),
                'mean_height': np.mean(heights),
                'std_width': np.std(widths),
                'std_height': np.std(heights),
                'min_width': np.min(widths),
                'max_width': np.max(widths),
                'min_height': np.min(heights),
                'max_height': np.max(heights)
            },
            'aspect_ratio': {
                'mean': np.mean(np.array(widths) / np.array(heights)),
                'std': np.std(np.array(widths) / np.array(heights))
            },
            'file_size': {
                'mean_mb': np.mean(sizes),
                'std_mb': np.std(sizes),
                'min_mb': np.min(sizes),
                'max_mb': np.max(sizes)
            }
        }
        
        return stats

    @staticmethod
    def get_training_transforms() -> transforms.Compose:
        """Get standard transformation pipeline for training.
        
        Returns:
            transforms.Compose: Standard transformation pipeline
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])