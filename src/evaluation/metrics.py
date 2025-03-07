from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from cleanfid import fid
import clip
from PIL import Image
import torch.nn.functional as F
import tempfile

__all__ = ["BaseMetric", "FIDScore", "CLIPScore", "MetricsTracker"]

class BaseMetric(ABC):
    """Base class for all evaluation metrics."""

    @abstractmethod
    def __init__(self):
        self.name: str
        
    @abstractmethod
    def compute(self, real_data: Any, generated_data: Any) -> float:
        """Compute metric between real and generated data."""
        pass
    
    @abstractmethod
    def update(self, real_data: Any, generated_data: Any):
        """Update metric state with new batch of data."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset metric state."""
        pass

class FIDScore(BaseMetric):
    """Fréchet Inception Distance (FID) score implementation.
    
    FID measures the similarity between two datasets by computing the
    Fréchet distance between two multivariate Gaussians fitted to
    feature representations of the Inception network.
    """
    
    def __init__(self, device: str = "cuda"):
        self.name = "fid"
        self.device = device
        self.real_images: List[np.ndarray] = []
        self.generated_images: List[np.ndarray] = []
        
    def _preprocess_images(self, images: torch.Tensor) -> np.ndarray:
        """Convert tensor images to numpy arrays in correct format."""
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
            # Convert from [-1, 1] to [0, 255]
            images = ((images + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
        return images
    
    def update(self, real_batch: torch.Tensor, generated_batch: torch.Tensor):
        """Add new batch of images to computation."""
        real_numpy = self._preprocess_images(real_batch)
        generated_numpy = self._preprocess_images(generated_batch)
        
        self.real_images.extend([img for img in real_numpy])
        self.generated_images.extend([img for img in generated_numpy])
    
    def compute(self, real_data: Any = None, generated_data: Any = None) -> float:
        """Compute FID score between accumulated real and generated images."""

        if real_data is not None and generated_data is not None:
            self.update(real_data, generated_data)
        
        if not self.real_images or not self.generated_images:
            raise ValueError("No images accumulated for FID computation")
        
        if len(self.real_images) > 0:
            print(f"FID Debug - Real images: shape={self.real_images[0].shape}, dtype={self.real_images[0].dtype}")
        if len(self.generated_images) > 0:
            print(f"FID Debug - Fake images: shape={self.generated_images[0].shape}, dtype={self.generated_images[0].dtype}")
            
        with tempfile.TemporaryDirectory() as real_dir, \
             tempfile.TemporaryDirectory() as fake_dir:
            
            real_dir, fake_dir = Path(real_dir), Path(fake_dir)
            
            for idx, (real, fake) in enumerate(zip(self.real_images, 
                                                 self.generated_images)):
                
                try:
                    if real.shape[0] == 3 and real.shape[2] != 3:  
                        real = np.transpose(real, (1, 2, 0))
                    
                    if fake.shape[0] == 3 and fake.shape[2] != 3:  
                        fake = np.transpose(fake, (1, 2, 0))
                    
                    if real.ndim == 2 or (real.ndim == 3 and real.shape[2] == 1):
                        real = np.repeat(real[:, :, np.newaxis], 3, axis=2) if real.ndim == 2 else np.repeat(real, 3, axis=2)
                    
                    if fake.ndim == 2 or (fake.ndim == 3 and fake.shape[2] == 1):
                        fake = np.repeat(fake[:, :, np.newaxis], 3, axis=2) if fake.ndim == 2 else np.repeat(fake, 3, axis=2)
                    
                    if idx < 3:
                        print(f"After transpose - Real shape: {real.shape}, Fake shape: {fake.shape}")
                    
                    Image.fromarray(real).save(real_dir / f"{idx}.png")
                    Image.fromarray(fake).save(fake_dir / f"{idx}.png")
                    
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
            
            try:
                fid_value = fid.compute_fid(str(real_dir), str(fake_dir))
                return fid_value
            except Exception as e:
                print(f"Error computing FID: {e}")
                # fallback value
                return float('inf')
    
    def reset(self):
        """Clear accumulated images."""
        self.real_images = []
        self.generated_images = []

    def _preprocess_images(self, images: torch.Tensor) -> List[np.ndarray]:
        """Convert tensor images to numpy arrays in correct format."""
        processed_images = []
        
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
            
            images = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            
            # Convert from PyTorch format (B, C, H, W) to numpy format (B, H, W, C)
            images_np = images.permute(0, 2, 3, 1).numpy()
            
            for img in images_np:
                processed_images.append(img)
        
        return processed_images

class CLIPScore(BaseMetric):
    """CLIP score implementation for text-image alignment evaluation.
    
    Measures how well generated images match their conditioning text
    using CLIP's image and text encoders.
    """
    
    def __init__(self, device: str = "cuda"):
        self.name = "clip_score"
        self.device = device
        
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.model.eval()
        
        self.image_features = []
        self.text_features = []
        
    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image tensor for CLIP model."""
        return F.interpolate(image, size=(224, 224), 
                           mode='bilinear', align_corners=False)
    
    def update(self, images: torch.Tensor, texts: List[str]):
        """Add new batch of images and texts to computation."""
        with torch.no_grad():
            # Preprocess and encode images
            processed_images = self._preprocess_image(images)
            batch_image_features = self.model.encode_image(processed_images)
            batch_image_features = batch_image_features / \
                                 batch_image_features.norm(dim=1, keepdim=True)
            
            # Encode text
            text_tokens = clip.tokenize(texts).to(self.device)
            batch_text_features = self.model.encode_text(text_tokens)
            batch_text_features = batch_text_features / \
                                batch_text_features.norm(dim=1, keepdim=True)
            
            self.image_features.append(batch_image_features)
            self.text_features.append(batch_text_features)
    
    def compute(self, images: torch.Tensor = None, 
                texts: List[str] = None) -> float:
        """Compute CLIP score between accumulated images and texts."""
        if images is not None and texts is not None:
            self.update(images, texts)
            
        if not self.image_features or not self.text_features:
            raise ValueError("No features accumulated for CLIP score computation")
            
        image_features = torch.cat(self.image_features)
        text_features = torch.cat(self.text_features)
        
        similarity = (100.0 * image_features @ text_features.T).mean()
        
        return similarity.item()
    
    def reset(self):
        """Clear accumulated features."""
        self.image_features = []
        self.text_features = []

class MetricsTracker:
    """Tracks and computes multiple metrics during evaluation."""
    
    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = {metric.name: metric for metric in metrics}
    
    def update(self, real_batch: torch.Tensor, generated_batch: torch.Tensor, 
               texts: List[str] = None):
        """Update all metrics with new batch."""
        for metric in self.metrics.values():
            if isinstance(metric, CLIPScore) and texts is not None:
                metric.update(generated_batch, texts)
            else:
                metric.update(real_batch, generated_batch)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics and return results."""
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.compute()
            except Exception as e:
                print(f"Error computing {name}: {e}")
        return results
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()