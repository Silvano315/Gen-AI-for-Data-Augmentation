import torch
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
from typing import List, Dict, Union, Optional
from pathlib import Path
import json
import re
from tqdm import tqdm

class TextVariationGenerator:
    """
    Generates variations of image captions using Flan-T5 model.
    
    This class handles the loading of a text-to-text model and provides
    methods for generating, saving, and managing text variations of 
    existing image captions.
    
    Attributes:
        device (torch.device): Device to run the model on
        model (AutoModelForSeq2SeqLM): Text generation model
        tokenizer (T5TokenizerFast): Tokenizer for the model
        variations_cache (Dict): Dictionary to store generated variations
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize the text variation generator.
        
        Args:
            model_name (str): Name of the T5 model to use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)  
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.variations_cache = {}
        
    def _extract_breed(self, caption: str) -> Optional[str]:
        """
        Extract breed information from caption.
        
        Args:
            caption (str): Caption that may contain breed information
            
        Returns:
            Optional[str]: Extracted breed or None if not found
        """
        match = re.search(r"This is an? ([^\.]+)\.", caption)
        if match:
            return match.group(1)
        return None
    
    def generate_variations(self, 
                        caption: str,
                        num_variations: int = 3,
                        temperature: float = 0.8,
                        preserve_breed: bool = True,
                        max_length: int = 100,
                        prompt_type: str = "standard",
                        max_tries: int = 1) -> List[str]:
        """
        Generate variations of a caption.
        
        Args:
            caption (str): Original caption to generate variations from
            num_variations (int): Number of variations to generate
            temperature (float): Sampling temperature (higher = more diverse)
            preserve_breed (bool): Whether to preserve breed information
            max_length (int): Maximum length of generated variations
            prompt_type (str): Type of prompt to use ('standard', 'specific', or 'few-shot')
            max_tries (int): Number of attempts to generate variations (useful for higher diversity)
            
        Returns:
            List[str]: List of generated variations
        """
        breed = None
        animal_type = "pet"
        
        if preserve_breed:
            breed = self._extract_breed(caption)
            
            if "cat" in caption.lower():
                animal_type = "cat"
            elif "dog" in caption.lower():
                animal_type = "dog"
                
            caption = re.sub(r"\s*-\s*This is an? [^\.]+\.", "", caption)
        
        if prompt_type == "specific":
            prompt = f"""Generate {num_variations} descriptions of this {animal_type} that vary significantly in:
                        - {animal_type.capitalize()}'s pose or position
                        - Background or environment
                        - Lighting conditions
                        - Actions the {animal_type} is performing

                        Original description: '{caption}'

                        Give completely different descriptions for each variation."""

        elif prompt_type == "few-shot":
            if animal_type == "cat":
                prompt = f"""Original: "A white cat sitting on a couch."
                            Variations:
                            1. A white cat playing with a toy mouse on the kitchen floor.
                            2. A white cat stretching lazily in a sunny garden.
                            3. A white cat curled up asleep on a blue blanket.

                            Original: "{caption}"
                            Variations:"""
            else:
                prompt = f"""Original: "A brown dog standing in the yard."
                            Variations:
                            1. A brown dog running excitedly at the beach, kicking up sand.
                            2. A brown dog resting under a tree in a park on a sunny day.
                            3. A brown dog playing with a ball in a living room with a fireplace.

                            Original: "{caption}"
                            Variations:"""
        else:
            prompt = f"Generate {num_variations} different descriptions of this {animal_type}: '{caption}'"
        
        all_variations = []
        
        for _ in range(max_tries):
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=num_variations,
                num_beams=num_variations
            )
            
            variations = [self.tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
            all_variations.extend(variations)
        
        # Post-process
        processed_variations = []
        for variation in all_variations:
            # Remove numbering
            variation = re.sub(r"^\d+[\.\)]\s*", "", variation)
            
            # Remove any "Variation:" prefixes
            variation = re.sub(r"^Variation:?\s*", "", variation)
            
            # Add breed information
            if preserve_breed and breed:
                if not re.search(rf"This is an? {breed}\.", variation):
                    variation = f"{variation} - This is a {breed}."
                    
            processed_variations.append(variation)
        
        # Remove duplicates
        unique_variations = []
        for var in processed_variations:
            if var not in unique_variations:
                unique_variations.append(var)
        
        return unique_variations[:num_variations]
    
    def process_caption_file(self, 
                            caption_file: Union[str, Path],
                            output_file: Optional[Union[str, Path]] = None,
                            variations_per_caption: int = 3,
                            class_balancing: bool = True,
                            target_per_class: int = 150,
                            min_variations: int = 1,
                            max_variations: int = 5,
                            prompt_type: str = "standard",
                            temperature: float = 0.8) -> Dict[str, List[str]]:
        """
        Process a caption file to generate variations with optional class balancing.
        
        Args:
            caption_file (Union[str, Path]): Path to JSON file with captions
            output_file (Optional[Union[str, Path]]): Path to save variations
            variations_per_caption (int): Base number of variations per caption
            class_balancing (bool): Whether to balance classes by generating more 
                                variations for underrepresented classes
            target_per_class (int): Target number of examples per class
            min_variations (int): Minimum variations to generate per caption
            max_variations (int): Maximum variations to generate per caption
            prompt_type (str): Type of prompt to use ('standard', 'specific', or 'few-shot')
            temperature (float): Temperature for generation (higher = more diverse)
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping original captions to variations
        """
        with open(caption_file, 'r') as f:
            captions = json.load(f)
        
        breed_counter = {}
        if class_balancing:
            for img_path, caption in captions.items():
                breed = self._extract_breed(caption)
                if breed:
                    breed_counter[breed] = breed_counter.get(breed, 0) + 1
        
        results = {}
        
        for img_path, caption in tqdm(captions.items(), desc="Generating variations"):
            breed = self._extract_breed(caption) if class_balancing else None
            
            # Determine number of variations based on class frequency
            num_vars = variations_per_caption
            if class_balancing and breed and breed in breed_counter:
                current_count = breed_counter[breed]
                if current_count < target_per_class:
                    # Scale the number of variations based on how underrepresented the class is
                    scarcity_factor = 1.0 - (current_count / target_per_class)
                    additional_vars = int(scarcity_factor * (max_variations - min_variations))
                    num_vars = min_variations + additional_vars
                else:
                    num_vars = min_variations
            
            variations = self.generate_variations(
            caption, 
            num_variations=num_vars,
            preserve_breed=True,
            prompt_type=prompt_type,
            temperature=temperature
            )
            
            results[img_path] = variations
            
            self.variations_cache[caption] = variations
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def select_balanced_subset(self,
                              caption_file: Union[str, Path],
                              variations_file: Union[str, Path],
                              output_file: Union[str, Path],
                              target_per_class: int = 150) -> Dict[str, str]:
        """
        Select a balanced subset of captions and variations to use for image generation.
        
        Args:
            caption_file (Union[str, Path]): Path to original captions file
            variations_file (Union[str, Path]): Path to generated variations file
            output_file (Union[str, Path]): Path to save selected captions
            target_per_class (int): Target number of examples per class
            
        Returns:
            Dict[str, str]: Selected caption subset
        """
        with open(caption_file, 'r') as f:
            original_captions = json.load(f)
        
        with open(variations_file, 'r') as f:
            variations = json.load(f)
        
        breed_to_images = {}
        for img_path, caption in original_captions.items():
            breed = self._extract_breed(caption)
            if breed:
                if breed not in breed_to_images:
                    breed_to_images[breed] = []
                breed_to_images[breed].append((img_path, caption, True))
        
        for img_path, var_list in variations.items():
            orig_caption = original_captions.get(img_path)
            if not orig_caption:
                continue
                
            breed = self._extract_breed(orig_caption)
            if not breed:
                continue
                
            for var in var_list:
                var_key = f"{img_path}_var_{len(breed_to_images[breed])}"
                breed_to_images[breed].append((var_key, var, False))
        
        selected_captions = {}
        
        for breed, items in breed_to_images.items():
            originals = [item for item in items if item[2]]
            variations = [item for item in items if not item[2]]
            
            needed = max(0, target_per_class - len(originals))
            selected_vars = variations[:needed] if needed > 0 else []
            
            for img_path, caption, _ in originals + selected_vars:
                selected_captions[img_path] = caption
        
        with open(output_file, 'w') as f:
            json.dump(selected_captions, f, indent=2)
        
        return selected_captions
    
    def save_variations(self, save_path: Union[str, Path]):
        """
        Save generated variations to JSON file.
        
        Args:
            save_path (Union[str, Path]): Path to save JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.variations_cache, f, indent=2)
    
    def load_variations(self, load_path: Union[str, Path]):
        """
        Load variations from JSON file.
        
        Args:
            load_path (Union[str, Path]): Path to JSON file
        """
        with open(load_path, 'r') as f:
            self.variations_cache = json.load(f)