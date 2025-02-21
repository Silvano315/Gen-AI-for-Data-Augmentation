import json
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict

class PetDatasetWithCaptions(Dataset):
    """Custom dataset that combines Oxford-IIIT Pet images with their captions."""
    
    def __init__(self, image_paths: List[str], caption_dict: Dict[str, str], transform=None):
        self.image_paths = image_paths
        self.caption_dict = caption_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        caption = self.caption_dict[image_path]
        
        return image, caption