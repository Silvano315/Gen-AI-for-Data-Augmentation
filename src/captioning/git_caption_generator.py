import torch
import numpy as np
from pathlib import Path
from PIL import Image
from textwrap import wrap
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import List, Dict, Optional, Union
import json

class GITCaptionGenerator:
    """Handles image captioning using GIT (Generative Image-to-text) model.
    
    This class manages the loading of the GIT model and provides methods
    for generating, saving, and visualizing image captions.
    
    Attributes:
        device (torch.device): Device to run the model on
        model (AutoModelForCausalLM): GIT model
        processor (AutoProcessor): GIT processor
        captions_cache (Dict): Dictionary to store generated captions
    """

    def __init__(self, model_name: str = "microsoft/git-base-coco"):
        """Initialize the caption generator.
        
        Args:
            model_name (str): Name of the GIT model to use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.captions_cache = {}

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or convert from PIL Image.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Image.Image: PIL Image object
        """
        if isinstance(image, (str, Path)):
            return Image.open(str(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError("Image must be a path or PIL Image")

    def generate_caption(self, 
                        image: Union[str, Path, Image.Image],
                        label: Optional[str] = None,
                        max_length: int = 50,
                        num_beams: int = 5) -> str:
        """Generate caption for a single image.
        
        Args:
            image: Image to caption (path or PIL Image)
            label: image's label to add breed
            max_length: Maximum length of generated caption
            num_beams: Number of beams for beam search
            
        Returns:
            str: Generated caption
        """
        img = self._load_image(image)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs.pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                min_length=5,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0
            )
            
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Add breed information
        if label:
            caption = f"{caption} - This is a {label}."

        if isinstance(image, (str, Path)):
            self.captions_cache[str(image)] = caption

        return caption

    def process_batch(self, 
                     image_paths: List[Union[str, Path]],
                     labels: Optional[List[str]] = None,
                     batch_size: int = 8) -> Dict[str, str]:
        """Generate captions for a batch of images.
        
        Args:
            image_paths: List of image paths
            labels: List of image labels
            batch_size: Size of batches for processing
            
        Returns:
            Dict[str, str]: Dictionary mapping image paths to captions
        """
        results = {}

        if labels is None:
            labels = [None] * len(image_paths)

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i : i+batch_size]
            batch_labels = labels[i:i + batch_size]

            for path, label in zip(batch_paths, batch_labels):
                caption = self.generate_caption(path, label=label)
                results[str(path)] = caption
        
        self.captions_cache.update(results)
        return results

    def save_captions(self, save_path: Union[str, Path]):
        """Save generated captions to JSON file.
        
        Args:
            save_path: Path to save JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self.captions_cache, f, indent=2)

    def load_captions(self, load_path: Union[str, Path]):
        """Load captions from JSON file.
        
        Args:
            load_path: Path to JSON file
        """
        with open(load_path, 'r') as f:
            self.captions_cache = json.load(f)

    def visualize_captions(self, 
                          num_samples: int = 4,
                          image_paths: Optional[List[Union[str, Path]]] = None,
                          wrap_width: int = 40) -> None:
        """Visualize images with their generated captions.
        
        Args:
            num_samples: Number of samples to visualize
            image_paths: Optional specific images to visualize
            wrap_width: Maximum number of characters per line in captions.
            
        Returns:
            None
        """
        if not self.captions_cache:
            raise ValueError("No captions available. Generate some captions first.")

        if image_paths is None:
            image_paths = np.random.choice(list(self.captions_cache.keys()),
                                          min(num_samples, len(self.captions_cache)),
                                          replace=False)

        n_cols = 2
        n_rows = (len(image_paths) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, path in enumerate(image_paths):
            img = Image.open(path).convert("RGB")
            img_array = np.array(img)

            axes[idx].imshow(img_array)
            axes[idx].axis('off')

            caption = self.captions_cache[str(path)]
            wrapped_caption = "\n".join(wrap(caption, wrap_width))
            axes[idx].set_title(wrapped_caption, fontsize=10)

        for ax in axes[len(image_paths):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()