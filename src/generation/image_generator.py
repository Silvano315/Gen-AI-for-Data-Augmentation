import torch
import torch.nn as nn
import torch.functional as F
import torch.optim
import torchvision.models as models
from transformers import BlipProcessor, BlipForImageTextRetrieval
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class GANConfig:
    """Configuration for Conditional GAN training and architecture.
    
    Attributes:
        latent_dim: Dimension of noise vector
        caption_dim: Dimension of caption embeddings from BLIP
        image_size: Size of generated images (assumed square)
        num_channels: Number of channels in generated images
        generator_features: Base number of features in generator
        learning_rate: Learning rate for both networks
        beta1: Beta1 parameter for Adam optimizer
        beta2: Beta2 parameter for Adam optimizer
    """
    latent_dim: int = 100
    caption_dim: int = 768
    image_size: int = 128
    num_channels: int = 3
    generator_features: int = 64
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999

class Generator(nn.Module):
    """
    Conditional Generator network.
    
    Takes noise vector and caption embedding as input, generates images.
    Uses transposed convolutions with batch normalization and ReLU activations.
    """

    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config

        self.project = nn.Sequential(
            nn.Linear(config.latent_dim + config.caption_dim,
                      config.generator_features * 16 * 4 * 4),
            nn.BatchNorm1d(config.generator_features * 16 * 4 * 4),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(config.generator_features * 16, 
                              config.generator_features * 8,
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config.generator_features * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(config.generator_features * 8,
                              config.generator_features * 4,
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config.generator_features * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(config.generator_features * 4,
                              config.generator_features * 2,
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config.generator_features * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(config.generator_features * 2,
                              config.generator_features,
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config.generator_features),
            nn.ReLU(True),

            nn.ConvTranspose2d(config.generator_features,
                              config.num_channels,
                              kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor, caption_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass of generator.
        
        Args:
            noise: Random noise vector of shape (batch_size, latent_dim)
            caption_embedding: Caption embeddings of shape (batch_size, caption_dim)
            
        Returns:
            Generated images of shape (batch_size, num_channels, image_size, image_size)
        """

        # Concatenate noise and caption embedding
        x = torch.concat([noise, caption_embedding], dim=1)

        x = self.project(x)
        x = x.view(-1, self.config.generator_features * 16, 4, 4)
        
        # Generate image through transposed convolutions
        return self.main(x)
    

class Discriminator(nn.Module):
    """
    Conditional Discriminator network using pretrained ResNet18.
    
    Takes images and caption embeddings as input, outputs real/fake predictions.
    Uses frozen ResNet18 for feature extraction followed by conditional classification.
    """
    
    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config

        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.caption_projection = nn.Sequential(
            nn.Linear(config.caption_dim, num_features),
            nn.BatchNorm1d(num_features),
            nn.LeakyReLU(0.2, True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.BatchNorm1d(num_features),
            nn.LeakyReLU(0.2, True),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor, caption_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of discriminator.
        
        Args:
            image: Input images of shape (batch_size, num_channels, image_size, image_size)
            caption_embedding: Caption embeddings of shape (batch_size, caption_dim)
            
        Returns:
            Probability predictions of shape (batch_size, 1)
        """

        image_features = self.resnet(image)

        caption_features = self.caption_projection(caption_embedding)

        combined_features = torch.cat([image_features, caption_features], dim = 1)

        return self.classifier(combined_features)
    

class ConditionalGAN(nn.Module):
    """Main Conditional GAN class handling training and inference.
    
    Combines Generator and Discriminator networks with training logic.
    """
    
    def __init__(self, config: GANConfig, blip_model_name: str = "Salesforce/blip-image-captioning-base"):
        super(ConditionalGAN, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        self.blip_model = BlipForImageTextRetrieval.from_pretrained(
                                "Salesforce/blip-image-captioning-base", 
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                            ).to(self.device)

        for param in self.blip_model.parameters():
            param.requires_grad = False

        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)
        
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )

        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.g_optimizer,
            T_max=200,  
            eta_min=1e-6
        )
        self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer,
            T_max=200,
            eta_min=1e-6
        )
        
        self.criterion = nn.BCELoss()

    def scheduler_step(self):
        """
        Step the learning rate schedulers at the end of each epoch
        """
        self.g_scheduler.step()
        self.d_scheduler.step()

    def get_caption_embeddings(self, captions) -> torch.Tensor:
        """Convert text captions to embeddings using BLIP.
        
        Args:
            captions: List or tuple of caption strings
            
        Returns:
            Tensor of caption embeddings (batch_size, caption_dim)
        """

        if isinstance(captions, tuple):
            captions = list(captions)

        inputs = self.blip_processor(text=captions, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            text_features = self.blip_model.text_encoder(**inputs).last_hidden_state  

            sentence_embeddings = text_features[:, 0, :]

        return sentence_embeddings
    
    def generate_from_text(self, caption_text: str, seed=None):
        """Generate an image from a text description.
        
        Args:
            caption_text: Text description of the image to generate
            seed: Optional random seed for reproducibility
        
        Returns:
            Generated image tensor
        """
        self.generator.eval()
        
        with torch.no_grad():
            caption_embedding = self.get_caption_embeddings([caption_text])
            
            if seed is not None:
                torch.manual_seed(seed)
            noise = torch.randn(1, self.config.latent_dim, device=self.device)
            
            image = self.generator(noise, caption_embedding)
        
        return image[0]
    

    def train_step(self, 
                  real_images: torch.Tensor,
                  caption_embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Single training step for both networks.
        
        Args:
            real_images: Batch of real images
            caption_embeddings: Corresponding caption embeddings
            
        Returns:
            Dictionary containing generator and discriminator losses
        """

        batch_size = real_images.size(0)
        real_label = torch.ones(batch_size, 1).to(self.device)
        fake_label = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real images
        d_real_output = self.discriminator(real_images, caption_embeddings)
        d_real_loss = self.criterion(d_real_output, real_label)
        
        # Fake images
        noise = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        fake_images = self.generator(noise, caption_embeddings)
        d_fake_output = self.discriminator(fake_images.detach(), caption_embeddings)
        d_fake_loss = self.criterion(d_fake_output, fake_label)
        
        # Loss Discriminator
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        g_output = self.discriminator(fake_images, caption_embeddings)
        g_loss = self.criterion(g_output, real_label)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "d_lr": self.d_optimizer.param_groups[0]['lr'],  
            "g_lr": self.g_optimizer.param_groups[0]['lr'] 
        }
    
    def generate(self, 
                caption_embeddings: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate images from caption embeddings.
        
        Args:
            caption_embeddings: Caption embeddings to condition on
            noise: Optional noise vectors (random if not provided)
            
        Returns:
            Generated images
        """

        self.generator.eval()
        
        with torch.no_grad():
            batch_size = caption_embeddings.size(0)
            if noise is None:
                noise = torch.randn(batch_size, self.config.latent_dim).to(self.device)
            images = self.generator(noise, caption_embeddings)
            
        self.generator.train()
        return images