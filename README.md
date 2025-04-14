# Gen AI for Data Augmentation

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methods](#methods)
- [Results](#rusults)
- [Key Insights](#key-insights)
- [EXTRA: Conditional GAN](#extra-conditional-gan)
- [How to Run](#how-to-run)


## Project Overview

This repository is the ninth project of the master's degree in AI Engineering with [Profession AI](https://profession.ai), all the credits for the requests and idea go to this team. 

The ability to accurately and timely recognize critical objects and behaviors in images is limited by the training datasets currently available, which do not fully represent the variability and complexity of real-world situations.

Benefits:
- Expanding the dataset using advanced Data Augmentation techniques will improve the accuracy of the image recognition system.
- By automating the process of generating new data through the creation of varied images and texts, operational efficiency will be optimized.
- Using advanced deep learning and data generation techniques with Generative AI will drive innovation.

Project steps and information:
* Dataset
    - Use the OxfordIIITPet dataset from PyTorch as the basis for the project to improve the image recognition system.
* Image Captioning and Data Generation
    - Apply image captioning to create initial descriptions of the images.
    - Then, use generative models to create variants of the images and texts, thus enriching the dataset with artificially generated data.
* Model Training:
    - Train an image recognition model using the extended dataset, evaluating the quality of the data produced and comparing the performance of the model on the reduced and augmented datasets.
* Performance Evaluation:
    - Measure accuracy, precision, recall and other performance metrics to compare the trained model on both datasets.
* Results Analysis:
    - Comment on the differences in performance and the effectiveness of data augmentation techniques in improving the accuracy of the model in real-world critical infrastructure security contexts.

## Dataset

The project utilizes the Oxford-IIIT Pet Dataset, which contains images of 37 different pet breeds (both cats and dogs). Key characteristics:

* Total of 7,349 images in the dataset
* 37 different pet categories with roughly 200 images per class
* Each class has a variable number of images
* Images contain variations in scale, pose, and lighting

The dataset was loaded and processed using a custom modular data handler class ([PetDatasetHandler](/src/data/dataset.py)), allowing for:

* Easy dataset exploration and statistics generation
* Visualization of class distribution
* Automatic transformation and preprocessing

## Methods

The project implements a modular, pipeline-based approach to generate and leverage augmented data:

### 1. Image Captioning
Multiple state-of-the-art captioning models from Hugging Face were evaluated:

* BLIP (Bootstrapping Language-Image Pre-training)
* BLIP-2
* GIT (Generative Image-to-Text Transformer)

After extensive comparison, the GIT model was selected for its optimal balance of descriptive quality and computational efficiency. The implementation includes:

* A dedicated GITCaptionGenerator class for systematic image captioning
* Efficient batch processing for generating captions across the dataset
* Caching system to avoid redundant caption generation

### 2. Text Variation Generation
Using Flan-T5 from Hugging Face, the project implements sophisticated text variation techniques:

* Various prompting strategies were tested and compared (zero-shot, few-shot, chain-of-thought)
* Advanced few-shot prompting produced the most diverse and accurate caption variations
* Custom temperature and sampling parameters for controllable diversity
* Class-balanced generation to ensure uniform distribution across breeds

The TextVariationGenerator class provides:

* Multiple prompt templates
* Batch processing capabilities
* Configurable generation parameters

### 3. Image Generation with Diffusion Models
The project features advanced diffusion model implementation with LoRA (Low-Rank Adaptation) fine-tuning:

* Evaluation of multiple diffusion models (Stable Diffusion v1.5, v2.1, SDXL, Kandinsky, FLUX)
* Fine-tuning of Stable Diffusion v1.5 using LoRA to specialize in pet breeds
* Custom dataset preparation for efficient LoRA training
* CLIP-based evaluation metrics for generated image quality assessment

The implementation includes:

* A comprehensive DiffusionModelManager for model loading, fine-tuning and inference
* Generation of balanced synthetic datasets based on class distribution
* Quality filtering and post-processing

### 4. Classification and Evaluation
Three distinct approaches were tested to evaluate augmentation effectiveness:

1. Original data only, no augmentation
2. Original data with traditional augmentation techniques
3. Original data + generated images (via LoRA fine-tuned diffusion)

Transfer learning with ResNet50 was implemented with:

* Optional custom classifier heads
* Comprehensive evaluation metrics
* Early stopping and learning rate scheduling
* Detailed visualization of training progress and results

## Results

The classification experiments yielded valuable insights into data augmentation effectiveness, you can see results and discussion directly in the [notebook](/GenAI_project.ipynb)

Key findings:

* The model trained on only original data showed the highest raw performance
* However, the models with augmentation (especially with generated images) showed more stable performance across classes
* The generative augmentation approach provided better performance on underrepresented classes
* Test set performance indicated better generalization capabilities with the augmented datasets
* Images difficult to classify were usually the breed of cats easily confused as Abyssinia and Sphinx

## Key Insights

1. Modular Pipeline Design: The project demonstrates a highly modular approach to data augmentation, with dedicated components for each phase (captioning, variation, generation, classification).
2. Hugging Face Integration: Extensive use of Hugging Face models (GIT, Flan-T5, Stable Diffusion) showcases the power of leveraging pre-trained models for specialized tasks.
3. LoRA Fine-tuning: The successful application of LoRA for efficient adaptation of diffusion models highlights an optimal approach for specialized image generation with limited computational resources.
4. Prompting Techniques: Various prompt engineering approaches were tested, with few-shot prompting yielding the best results for text variation generation.
5. Dataset Balance: While raw performance metrics were slightly lower with augmented data, the balanced nature of the resulting dataset provides more uniform performance across classes.
6. CLIP Evaluation: Using CLIP for evaluating generated image quality provides an objective metric for assessing synthetic data.
7. Tradeoffs: The project illustrates important tradeoffs between model complexity, computational requirements, and augmentation quality.

## EXTRA: Conditional GAN

In addition to the diffusion model approach, I initially explored using a Conditional GAN architecture for generating pet images based on text descriptions. This approach offered a potentially faster inference time compared to diffusion models, with the following implementation details:
1. Architecture
    * Generator: A deep convolutional network with transposed convolutions, taking both noise vectors and BLIP-encoded caption embeddings as input
    * Discriminator: A custom network with spectral normalization for improved stability, conditioning both on images and caption embeddings
    * Text Encoder: Integrated BLIP model (Salesforce/blip-image-captioning-base) to convert captions to meaningful embeddings
2. Training Framework
    * A comprehensive training loop with metrics tracking and visualization
    * Adaptive learning rate schedulers using CosineAnnealingLR
    * Robust checkpointing and early stopping
    * CLIP-based evaluation metrics to assess image-text alignment
    * FID calculation to measure the quality of generated images compared to real ones
3. Advanced Features
    * Caption-guided image generation
    * Noise and caption space interpolation for smooth transitions between samples
    * Batch processing for efficient training
    * Visualization tools for generated samples

This GAN-based approach was fully implemented with an extensive set of tools for both training and evaluation. However, despite the elegant design, it required substantial computational resources beyond what was available for this project. The diffusion model with LoRA fine-tuning ultimately proved to be a more practical approach given the compute constraints, while still producing high-quality, text-conditioned images.

## How to Run

Clone the repository:
```bash
git clone https://github.com/Silvano315/Gen-AI-for-Data-Augmentation.git
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook GenAI_project.ipynb
```

The project's modular structure includes:

* src/data/: Dataset handling and preprocessing
* src/captioning/: Image captioning modules
* src/generation/: Text and image generation components
* src/models/: Classification model implementations
* src/training/: Training and evaluation utilities
* src/visualization/: Visualization tools

Note: To run the diffusion model fine-tuning, you'll need a GPU with at least 16GB of VRAM or access to Google Colab with GPU runtime.