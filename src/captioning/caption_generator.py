import torch
import json
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from plotly.subplots import make_subplots
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Dict, Optional, Union

class CaptionGenerator:
    """Handles image captioning using BLIP model.
    
    This class manages the loading of the BLIP model and provides methods
    for generating, saving, and visualizing image captions.
    
    Attributes:
        device (torch.device): Device to run the model on
        model (BlipForConditionalGeneration): BLIP model
        processor (BlipProcessor): BLIP processor
        captions_cache (Dict): Dictionary to store generated captions
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """Initialize the caption generator.
        
        Args:
            model_name (str): Name of the BLIP model to use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
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
                        max_length: int = 30,
                        num_beams: int = 4) -> str:
        """Generate caption for a single image.
        
        Args:
            image: Image to caption (path or PIL Image)
            max_length: Maximum length of generated caption
            num_beams: Number of beams for beam search
            
        Returns:
            str: Generated caption
        """
        img = self._load_image(image)
        inputs = self.processor(img, return_tensors = "pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length = max_length,
                num_beams = num_beams,
                early_stopping = True
            )

        caption = self.processor.decode(outputs[0], skip_special_tokens = True)

        if isinstance(image, (str, Path)):
            self.captions_cache[str(image)] = caption

        return caption

    def process_batch(self, 
                     image_paths: List[Union[str, Path]],
                     batch_size: int = 16) -> Dict[str, str]:
        """Generate captions for a batch of images.
        
        Args:
            image_paths: List of image paths
            batch_size: Size of batches for processing
            
        Returns:
            Dict[str, str]: Dictionary mapping image paths to captions
        """
        results = {}

        for i in tqdm(range(0, len(image_paths), batch_size), desc = "Processing batches"):
            batch_paths = image_paths[i : i+batch_size]

            for path in batch_paths:
                caption = self.generate_caption(path)
                results[str(path)] = caption
        
        self.captions_cache.update(results)
        return results

    def save_captions(self, save_path: Union[str, Path]):
        """Save generated captions to JSON file.
        
        Args:
            save_path: Path to save JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents = True, exist_ok = True)

        with open(save_path, "w") as f:
            json.dump(self.captions_cache, f, indent = 2)

    def load_captions(self, load_path: Union[str, Path]):
        """Load captions from JSON file.
        
        Args:
            load_path: Path to JSON file
        """
        with open(load_path, 'r') as f:
            self.captions_cache = json.load(f)

    def visualize_captions(self, 
                          num_samples: int = 4,
                          image_paths: Optional[List[Union[str, Path]]] = None) -> go.Figure:
        """Visualize images with their generated captions.
        
        Args:
            num_samples: Number of samples to visualize
            image_paths: Optional specific images to visualize
            
        Returns:
            go.Figure: Plotly figure with images and captions
        """
        if not self.captions_cache:
            raise ValueError("No captions available. Generate some captions first.")
            
        if image_paths is None:
            image_paths = np.random.choice(list(self.captions_cache.keys()),
                                            min(num_samples, len(self.captions_cache)),
                                            replace = False)
        
        n_cols = 2
        n_rows = (len(image_paths) + 1) // 2
        fig = make_subplots(rows = n_rows, cols = n_cols,
                            subplot_titles=[self.captions_cache[str(path)] for path in image_paths])
        
        for idx, path in enumerate(image_paths):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            img = Image.open(path).convert("RGB")
            img_array = np.array(img)

            fig.add_trace(
                go.Image(z = img_array),
                row = row, col = col
            )

        fig.update_layout(
            height=400 * n_rows,
            width=800,
            showlegend=False,
            title_text="Generated Captions"
        )
        
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig