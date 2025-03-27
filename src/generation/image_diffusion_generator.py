import torch
import json
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List, Dict, Union, Optional
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    AutoPipelineForText2Image, 
    FluxPipeline
)

class DiffusionModelManager:
    """
    Unified manager class for diffusion models that handles:
    - Zero-shot testing with different models
    - Dataset preparation for LoRA fine-tuning
    - LoRA fine-tuning process
    - Inference with fine-tuned models
    - Image generation from text variations
    
    This class centralizes all functionality related to diffusion models
    in a single interface for easier workflow management.
    """
    
    def __init__(
        self,
        base_models_dir: Optional[str] = None,
        output_dir: str = "diffusion_output",
        device: str = "cuda",
        default_model: str = "runwayml/stable-diffusion-v1-5"
    ):
        """
        Initialize the diffusion model manager.
        
        Args:
            base_models_dir: Directory to store cached models
            output_dir: Directory for outputs (images, logs, checkpoints)
            device: Device to use for inference/training (cuda or cpu)
            default_model: Default model ID to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = Path(base_models_dir) if base_models_dir else None
        self.default_model = default_model

        self.current_pipeline = None
        self.current_model_id = None
        
        (self.output_dir / "zero_shot").mkdir(exist_ok=True)
        (self.output_dir / "fine_tuned").mkdir(exist_ok=True)
        (self.output_dir / "datasets").mkdir(exist_ok=True)
        (self.output_dir / "lora_models").mkdir(exist_ok=True)
        
        print(f"DiffusionModelManager initialized with device: {self.device}")
        print(f"Output directory: {self.output_dir}")

    def test_diffusion_model(
        self,
        model_id: str,
        caption: str,
        num_images: int = 1,
        seed: Optional[int] = None,
        output_subdir: Optional[str] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        width: int = 512,
        height: int = 512,
        save_images: bool = True
    ) -> List[Image.Image]:
        """
        Test a diffusion model with a specific caption in zero-shot setting.
        
        Args:
            model_id: Hugging Face model ID
            caption: Text caption for image generation
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            output_subdir: Subdirectory to save images
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            width: Image width
            height: Image height
            save_images: Whether to save generated images
            
        Returns:
            List of generated PIL images
        """
        print(f"Testing model: {model_id}")

        if "xl" in model_id.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                use_safetensors=True, 
                variant="fp16"
            )
        elif "kandinsky" in model_id.lower():
            pipe = AutoPipelineForText2Image.from_pretrained(
                        "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
                    ).to("cuda")
        elif "flux" in model_id.lower():
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16 
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            )
        
        pipe = pipe.to("cuda")

        # Save memory
        pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()

        breed_info = ""
        if " - This is " in caption[0]:
            parts = caption[0].split(" - This is a ")
            cleaned_caption = parts[0]
            breed_info = parts[1].strip(".")
            prompt = f"A high-quality photo of a {breed_info}, {cleaned_caption}"
        else:
            prompt = f"A high-quality photo of {caption}"

        print(f"Prompt: {prompt}")

        images = []
        for i in range(num_images):
            generator = None
            if seed is not None:
                generator = torch.Generator(device = "cuda").manual_seed(seed + i)

            if "kandinsky" in model_id.lower():
                image = pipe(prompt, generator = generator).images[0]
            elif "flux" in model_id.lower():
                image = pipe(prompt,
                             guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            width=width,
                            height=height,
                            max_sequence_length=512
                            ).images[0]
            else:
                image = pipe(
                    prompt, 
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                ).images[0]
            
            images.append(image)

        if save_images:
            save_dir = self.output_dir / "zero_shot"
            if output_subdir:
                save_dir = save_dir / output_subdir
                
            save_dir.mkdir(parents=True, exist_ok=True)
            
            model_name = model_id.split("/")[-1]
            for i, img in enumerate(images):
                img_path = save_dir / f"{model_name}_{i}_seed{seed}.png"
                img.save(img_path)
                
                with open(save_dir / f"{model_name}_{i}_seed{seed}.txt", "w") as f:
                    f.write(prompt)

        fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
        if num_images == 1:
            axes = [axes]
        
        for i, img in enumerate(images):
            axes[i].imshow(np.array(img))
            axes[i].set_title(f"Image {i+1}")
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        return images

    def prepare_dataset(
        self,
        captions_file: Union[str, Path],
        images_dir: Union[str, Path],
        output_name: Optional[str] = None,
        max_samples_per_breed: Optional[int] = None,
        min_samples_per_breed: int = 3,
        target_total_samples: Optional[int] = None,
        class_field: str = "breed",
        resolution: int = 512
    ) -> Path:
        """
        Prepare a dataset for LoRA fine-tuning.
        
        Args:
            captions_file: Path to JSON file with captions
            images_dir: Directory containing images
            output_name: Name of output dataset directory
            max_samples_per_breed: Maximum samples per class
            min_samples_per_breed: Minimum samples per class
            target_total_samples: Target total number of samples
            class_field: Field name for class information extraction
            resolution: Target resolution for images
            
        Returns:
            Path to prepared dataset
        """
        if output_name:
            output_dir = self.output_dir / "datasets" / output_name
        else:
            timestamp = Path(captions_file).stem
            output_dir = self.output_dir / "datasets" / f"dataset_{timestamp}"
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(captions_file, "r") as f:
            captions = json.load(f)
                
        breed_samples = {}
        for img_path, caption in captions.items():
            if " - This is a " in caption:
                breed = caption.split(" - This is a ")[1].strip(".")

                if not breed:
                    continue

                if breed not in breed_samples:
                    breed_samples[breed] = []
                
                img_name = Path(img_path).name
                full_img_path = Path(images_dir) / img_name

                if full_img_path.exists():
                    breed_samples[breed].append((str(full_img_path), caption))
                else:
                    print(f"Warning: Image not found: {img_path}")
                    continue

        
        # Select a balanced subset
        selected_samples = []
        metadata = []
        
        for breed, samples in breed_samples.items():
            num_samples = min(
                len(samples),
                max_samples_per_breed if max_samples_per_breed else len(samples)
            )
            num_samples = max(num_samples, min_samples_per_breed)
            
            # Select random sample
            breed_selection = random.sample(samples, min(num_samples, len(samples)))
            selected_samples.extend(breed_selection)
        
        # Limit with target_total_samples
        if target_total_samples and len(selected_samples) > target_total_samples:
            random.shuffle(selected_samples)
            selected_samples = selected_samples[:target_total_samples]
        
        print(f"Selected {len(selected_samples)} samples from {len(breed_samples)} breed")
        
        # Create a metadata.jsonl as required by LoRA
        for i, (img_path, caption) in enumerate(selected_samples):
            dest_filename = f"image_{i:06d}.jpg"
            dest_path = output_dir / dest_filename
            #shutil.copy(img_path, dest_path)

            img = Image.open(img_path).convert("RGB")
            
            # Resize if needed
            if resolution:
                # Center crop to square while maintaining aspect ratio
                width, height = img.size
                min_dim = min(width, height)
                left = (width - min_dim) // 2
                top = (height - min_dim) // 2
                right = left + min_dim
                bottom = top + min_dim
                
                img = img.crop((left, top, right, bottom))
                img = img.resize((resolution, resolution), Image.LANCZOS)
                
            # Save image
            img.save(dest_path)
            
            metadata.append({
                "file_name": dest_filename,
                "text": caption
            })
        
        metadata_path = output_dir / "metadata.jsonl"
        with open(metadata_path, "w") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")
        
        print(f"Dataset prepared in {output_dir}")
        print(f"Metadata saved to: {metadata_path}")

        return output_dir

    def select_validation_prompts_from_variations(
        self, 
        variations_file: Union[str, Path], 
        num_prompts: int = 5, 
        seed: int = 42
    ) -> List[str]:
        """
        Select validation prompts from a variations file.
        
        Args:
            variations_file: Path to variations JSON file
            num_prompts: Number of prompts to select
            seed: Random seed for selection
            
        Returns:
            List of selected prompts
        """
        with open(variations_file, 'r') as f:
            variations = json.load(f)
        
        all_captions = []
        for variations_list in variations.values():
            all_captions.extend(variations_list)
        
        random.seed(seed)
        selected_prompts = random.sample(all_captions, min(num_prompts, len(all_captions)))
        
        return selected_prompts

    def run_lora_training(
        self,
        dataset_dir: Union[str, Path],
        output_name: Optional[str] = None,
        base_model: Optional[str] = None,
        resolution: int = 512,
        train_batch_size: int = 1,
        max_train_steps: int = 1000,
        learning_rate: float = 1e-4,
        validation_prompts: Optional[List[str]] = None,
        rank: int = 4
    ) -> Path:
        """
        Run LoRA fine-tuning on a prepared dataset.
        
        Args:
            dataset_dir: Directory with prepared dataset
            output_name: Name for the output directory
            base_model: Base model ID
            resolution: Image resolution
            train_batch_size: Batch size for training
            max_train_steps: Maximum training steps
            learning_rate: Learning rate
            validation_prompts: Prompts for validation
            rank: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            
        Returns:
            Path to fine-tuned model
        """
        dataset_dir = Path(dataset_dir)
        
        if output_name:
            output_dir = self.output_dir / "lora_models" / output_name
        else:
            timestamp = dataset_dir.name
            output_dir = self.output_dir / "lora_models" / f"lora_{timestamp}"
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if base_model is None:
            base_model = self.default_model
            
        print(f"Starting LoRA training with base model: {base_model}")
        print(f"Dataset: {dataset_dir}")
        print(f"Output directory: {output_dir}")

        # Correct diffusers version if you didn't run it before
        try:
            subprocess.run(["pip", "install", "-q", "git+https://github.com/huggingface/diffusers.git"])
            subprocess.run(["pip", "install", "-q", "accelerate", "transformers", "bitsandbytes", "datasets"])
        except Exception as e:
            print(f"Warning: Failed to install dependencies - {e}")
            
        # We need to locally download the most up-to-date version of train_text_to_image_lora.py
        script_path = Path.cwd() / "train_text_to_image_lora.py"
        try:
            subprocess.run([
                "wget", 
                "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py", 
                "-O", 
                str(script_path)
                # "train_text_to_image_lora.py"
            ])
        except Exception as e:
            print(f"Warning: Failed to download script - {e}")
            if not script_path.exists():
                raise RuntimeError("Could not download training script and no local script found.")
                
        if validation_prompts is None:
            validation_prompts = [
                "A high-quality photo of a dog",
                "A high-quality photo of a cat",
                "A close-up portrait of a pet"
            ]

        # Command to use train_text_to_image_lora.py
        cmd = [
            "accelerate", "launch",
            "train_text_to_image_lora.py",
            f"--pretrained_model_name_or_path={base_model}",
            f"--train_data_dir={dataset_dir}",
            f"--output_dir={output_dir}",
            f"--resolution={resolution}",
            "--center_crop",
            "--random_flip",
            f"--train_batch_size={train_batch_size}",
            "--gradient_accumulation_steps=4",
            "--gradient_checkpointing",
            "--mixed_precision=fp16",
            f"--max_train_steps={max_train_steps}",
            f"--learning_rate={learning_rate}",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            "--validation_epochs=100",
            f"--validation_prompt=\"{'; '.join(validation_prompts)}\"",
            "--seed=42",
            "--checkpointing_steps=500",
            f"--rank={rank}"
        ]

        print(f"Running command: {' '.join(cmd)}")
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            # Log output
            with open(output_dir / "training_log.txt", "w") as f:
                f.write(f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")
                
            if process.returncode != 0:
                print(f"Warning: Training process exited with code {process.returncode}")
                print(f"Error details: {process.stderr}")
            else:
                print(f"Training completed successfully!")
        except Exception as e:
            print(f"Error during training: {e}")
            raise
            
        return output_dir

    def load_lora_model(
        self,
        base_model_id: Optional[str] = None,
        lora_weights_path: Optional[str] = None,
        torch_dtype = torch.float16
    ):
        """
        Load a model with LoRA weights.
        
        Args:
            base_model_id: ID of the base model
            lora_weights_path: Path to LoRA weights
            torch_dtype: Data type for model loading
            
        Returns:
            self for method chaining
        """
        if base_model_id is None:
            base_model_id = self.default_model
            
        print(f"Loading base model: {base_model_id}")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype
        )
        
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        if lora_weights_path:
            print(f"Loading LoRA weights from: {lora_weights_path}")
            pipeline.unet.load_attn_procs(lora_weights_path)
            #pipeline.load_lora_adapter(lora_weights_path)
            print("LoRA weights loaded successfully!")
        
        pipeline.to(self.device)
        pipeline.enable_attention_slicing()
        
        self.current_pipeline = pipeline
        self.current_model_id = base_model_id
        
        return self

    def generate_images(
        self,
        prompts: Union[str, List[str]],
        output_subdir: Optional[str] = None,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
        save_images: bool = True,
        display_images: bool = True,
        batch_size: int = 1
    ) -> Dict[str, List[Image.Image]]:
        """
        Generate images using the current model pipeline.
        
        Args:
            prompts: Text prompts for image generation
            output_subdir: Subdirectory to save images
            num_images_per_prompt: Number of images per prompt
            guidance_scale: Guidance scale for classifier-free guidance
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            width: Image width
            height: Image height
            save_images: Whether to save generated images
            display_images: Whether to display generated images
            batch_size: Batch size for generation
            
        Returns:
            Dictionary mapping prompts to generated images
        """
        if self.current_pipeline is None:
            self.load_lora_model()
            
        if isinstance(prompts, str):
            prompts = [prompts]
            
        if save_images:
            if output_subdir:
                output_path = self.output_dir / "fine_tuned" / output_subdir
            else:
                output_path = self.output_dir / "fine_tuned" / "generated"
                
            output_path.mkdir(parents=True, exist_ok=True)
            
        results = {}
        all_images = []
        all_prompts = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                seed += 1
                
            print(f"Generating batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}...")
            
            batch_results = self.current_pipeline(
                batch_prompts,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                width=width,
                height=height
            )
            
            batch_images = batch_results.images
            
            for j, prompt in enumerate(batch_prompts):
                start_idx = j * num_images_per_prompt
                end_idx = start_idx + num_images_per_prompt
                prompt_images = batch_images[start_idx:end_idx]
                
                results[prompt] = prompt_images
                all_images.extend(prompt_images)
                all_prompts.extend([prompt] * num_images_per_prompt)
                
                if save_images:
                    for k, img in enumerate(prompt_images):
                        prompt_hash = abs(hash(prompt)) % 10000
                        img_path = output_path / f"gen_{prompt_hash}_{i}_{j}_{k}.png"
                        img.save(img_path)
                        
                        with open(output_path / f"gen_{prompt_hash}_{i}_{j}_{k}.txt", "w") as f:
                            f.write(prompt)
        
        if display_images and all_images:
            n_cols = min(5, len(all_images))
            n_rows = (len(all_images) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(n_cols * 5, n_rows * 6))
            
            for i, (image, prompt) in enumerate(zip(all_images, all_prompts)):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(np.array(image))
                plt.title(prompt[:50] + "..." if len(prompt) > 50 else prompt, fontsize=8)
                plt.axis("off")
                
            plt.tight_layout()
            plt.show()
            
        return results
    

    def generate_from_variations(
        self,
        variations_file: Union[str, Path],
        output_subdir: Optional[str] = None,
        num_samples: int = 5,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        random_variations: bool = True
    ) -> Dict[str, List[Image.Image]]:
        """
        Generate images from text variations in a file.
        
        Args:
            variations_file: JSON file with text variations
            output_subdir: Subdirectory for saving images
            num_samples: Number of variation prompts to use
            num_images_per_prompt: Number of images per prompt
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            random_variations: Whether to sample variations randomly
            
        Returns:
            Dictionary mapping prompts to generated images
        """
        with open(variations_file, "r") as f:
            variations_data = json.load(f)
            
        all_variations = []
        for img_path, variations in variations_data.items():
            all_variations.extend(variations)
            
        if random_variations:
            selected_variations = random.sample(all_variations, min(num_samples, len(all_variations)))
        else:
            selected_variations = all_variations[:num_samples]
            
        return self.generate_images(
            prompts=selected_variations,
            output_subdir=output_subdir or "variations",
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed
        )