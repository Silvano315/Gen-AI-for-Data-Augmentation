import json
import time
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Any
from src.utils.logging import GANLogger
from src.evaluation.metrics import MetricsTracker
from src.generation.image_generator import ConditionalGAN

class GANTrainer:
    """Main trainer class for Conditional GAN.
    
    Handles:
    - Training loop with callbacks
    - Metric computation
    - Logging
    - Checkpointing
    - Sample generation and visualization
    """
    
    def __init__(
        self,
        gan: ConditionalGAN,  
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metrics_tracker: MetricsTracker,  
        logger: GANLogger,  
        callbacks: List[Any] = None,
        device: str = "cuda"
    ):
        self.gan = gan
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metrics_tracker = metrics_tracker
        self.logger = logger
        self.callbacks = callbacks or []
        self.device = device
        
        self.current_epoch = 0
        self.current_step = 0
        self.stop_training = False
        
        self.gan.to(device)

    def _run_callbacks(self, hook: str, *args, **kwargs):
        """Execute all callbacks for a given hook."""
        for callback in self.callbacks:
            getattr(callback, hook)(self, *args, **kwargs)
            
    def _train_step(self, real_images: torch.Tensor, 
                   captions: List[str]) -> Dict[str, float]:
        """Execute single training step."""
        real_images = real_images.to(self.device)
        
        caption_embeddings = self.gan.get_caption_embeddings(captions)
        
        return self.gan.train_step(real_images, caption_embeddings)
    
    def evaluate(self, num_batches: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation loop."""
        self.gan.eval()
        self._run_callbacks('on_evaluate_begin')
        
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for i, (real_images, captions) in enumerate(self.val_dataloader):
                if num_batches and i >= num_batches:
                    break
                    
                real_images = real_images.to(self.device)
                caption_embeddings = self.gan.get_caption_embeddings(captions)
                
                noise = torch.randn(len(real_images), 
                                  self.gan.config.latent_dim,
                                  device=self.device)
                fake_images = self.gan.generate(caption_embeddings, noise)
                
                self.metrics_tracker.update(real_images, fake_images, captions)
        
        metrics = self.metrics_tracker.compute()
        
        self._run_callbacks('on_evaluate_end', metrics)
        self.gan.train()
        
        return metrics
        
    def train(
        self,
        num_epochs: int,
        eval_freq: int = 1,
        num_eval_batches: Optional[int] = None,
        sample_freq: Optional[int] = None,
        sample_dir: Optional[Path] = None
    ):
        """Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            eval_freq: Frequency of evaluation in epochs
            num_eval_batches: Number of batches to use for evaluation
            sample_freq: Frequency of sample generation in steps
            sample_dir: Directory to save generated samples
        """
        self._run_callbacks('on_train_begin')
        
        start_time = time.time()
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                if self.stop_training:
                    break
                    
                self.current_epoch = epoch
                epoch_start_time = time.time()
                self._run_callbacks('on_epoch_begin', epoch)
                
                pbar = tqdm(self.train_dataloader, 
                          desc=f"Epoch {epoch}/{num_epochs}")
                
                for batch_idx, (real_images, captions) in enumerate(pbar):
                    self._run_callbacks('on_batch_begin', 
                                      (real_images, captions))
                    
                    losses = self._train_step(real_images, captions)
                    
                    pbar.set_postfix(losses)
                    
                    self.logger.log_metrics(
                        losses,
                        self.current_step
                    )
                    
                    self._run_callbacks('on_batch_end', 
                                      (real_images, captions), 
                                      losses)
                    
                    if sample_freq and self.current_step % sample_freq == 0:
                        self.save_generated_samples(
                            num_samples=min(8, len(real_images)),
                            save_dir=sample_dir / f"step_{self.current_step}"
                        )
                    
                    self.current_step += 1
                    
                if epoch % eval_freq == 0:
                    self.logger.info(f"Running evaluation at epoch {epoch}")
                    metrics = self.evaluate(num_eval_batches)
                    self.logger.log_metrics(metrics, self.current_step)
                
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch} completed in {epoch_time:.2f} seconds"
                )
                
                self._run_callbacks('on_epoch_end', epoch)
                
                self.gan.scheduler_step()
                
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
        finally:
            training_time = time.time() - start_time
            self.logger.info(
                f"Training finished after {training_time:.2f} seconds"
            )
            
            self.logger.save_metrics()
            
            self._run_callbacks('on_train_end')
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load training state from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.gan.generator.load_state_dict(checkpoint['generator_state'])
        self.gan.discriminator.load_state_dict(checkpoint['discriminator_state'])
        
        self.gan.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.gan.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        
        self.gan.g_scheduler.load_state_dict(checkpoint['g_scheduler'])
        self.gan.d_scheduler.load_state_dict(checkpoint['d_scheduler'])
        
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        
        self.logger.info(
            f"Restored checkpoint from epoch {self.current_epoch}"
        )
        
        if 'best_metrics' in checkpoint:
            self.logger.best_metrics.update(checkpoint['best_metrics'])
        
    def save_generated_samples(self, 
                             num_samples: int,
                             save_dir: Path,
                             fixed_noise: Optional[torch.Tensor] = None):
        """Generate and save sample images.
        
        Args:
            num_samples: Number of samples to generate
            save_dir: Directory to save samples
            fixed_noise: Optional fixed noise for consistent samples
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.gan.eval()
        with torch.no_grad():
            real_images, captions = next(iter(self.val_dataloader))
            real_images = real_images[:num_samples]
            captions = captions[:num_samples]
            
            caption_embeddings = self.gan.get_caption_embeddings(captions)
            
            if fixed_noise is None:
                noise = torch.randn(num_samples, 
                                  self.gan.config.latent_dim,
                                  device=self.device)
            else:
                noise = fixed_noise
                
            fake_images = self.gan.generate(caption_embeddings, noise)
            
            for i, (real, fake, caption) in enumerate(zip(real_images, 
                                                        fake_images, 
                                                        captions)):
                save_image(
                    real,
                    save_dir / f"real_{i}.png",
                    normalize=True,
                    range=(-1, 1)
                )
                
                save_image(
                    fake,
                    save_dir / f"fake_{i}.png",
                    normalize=True,
                    range=(-1, 1)
                )
            
            with open(save_dir / "captions.json", 'w') as f:
                json.dump(
                    {f"sample_{i}": caption for i, caption in enumerate(captions)},
                    f,
                    indent=2
                )
        
        self.gan.train()
    
    def generate_interpolations(self,
                              num_steps: int,
                              save_dir: Path,
                              noise_interpolation: bool = True,
                              caption_interpolation: bool = True):
        """Generate interpolations between samples.
        
        Args:
            num_steps: Number of interpolation steps
            save_dir: Directory to save interpolation samples
            noise_interpolation: Whether to interpolate in noise space
            caption_interpolation: Whether to interpolate in caption space
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.gan.eval()
        with torch.no_grad():
            real_images, captions = next(iter(self.val_dataloader))
            real_1, real_2 = real_images[:2]
            caption_1, caption_2 = captions[:2]
            
            caption_emb_1 = self.gan.get_caption_embeddings([caption_1])[0]
            caption_emb_2 = self.gan.get_caption_embeddings([caption_2])[0]
            
            noise_1 = torch.randn(1, self.gan.config.latent_dim, 
                                device=self.device)
            noise_2 = torch.randn(1, self.gan.config.latent_dim, 
                                device=self.device)
            
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                
                if noise_interpolation:
                    noise = torch.lerp(noise_1, noise_2, alpha)
                    fake = self.gan.generate(
                        caption_emb_1.unsqueeze(0),
                        noise
                    )
                    save_image(
                        fake[0],
                        save_dir / f"noise_interp_{i}.png",
                        normalize=True,
                        range=(-1, 1)
                    )
                
                if caption_interpolation:
                    caption_emb = torch.lerp(caption_emb_1, 
                                           caption_emb_2,
                                           alpha)
                    fake = self.gan.generate(
                        caption_emb.unsqueeze(0),
                        noise_1
                    )
                    save_image(
                        fake[0],
                        save_dir / f"caption_interp_{i}.png",
                        normalize=True,
                        range=(-1, 1)
                    )
            
            with open(save_dir / "interpolation_captions.json", 'w') as f:
                json.dump({
                    "caption_1": caption_1,
                    "caption_2": caption_2
                }, f, indent=2)
        
        self.gan.train()