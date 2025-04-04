from typing import Dict, Any, List
import torch.nn as nn
import timm
from .classifier import BaselineCNN, TransferLearningModel

def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        model_config: Dictionary containing model configuration
            Required keys:
            - type: str, Model type ('baseline', 'transfer')
            - num_classes: int, Number of output classes
            For transfer learning models:
            - model_name: str, Name of the timm model to use
            - pretrained: bool, Whether to use pretrained weights
            - use_custom_classifier: bool, Whether to use custom classifier
            For baseline models:
            - input_channels: int, Number of input channels
            
    Returns:
        Instantiated model
    """
    model_type = model_config.pop('type')
    
    if 'num_classes' not in model_config:
        raise ValueError("num_classes must be specified in model_config")
    
    if model_type == 'baseline':
        if 'input_channels' not in model_config:
            model_config['input_channels'] = 3
        return BaselineCNN(**model_config)
    
    elif model_type == 'transfer':
        required_keys = ['model_name', 'pretrained', 'use_custom_classifier']
        missing_keys = [key for key in required_keys if key not in model_config]
        if missing_keys:
            raise ValueError(f"Missing required config keys for transfer learning: {missing_keys}")
        
        return TransferLearningModel(**model_config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_available_models() -> Dict[str, List[str]]:
    """
    Get a dictionary of available model types and specific models.
    
    Returns:
        Dictionary with model types as keys and lists of available models as values
    """
    return {
        'baseline': ['baseline_cnn'],
        'transfer': timm.list_models()
    }

def validate_model_config(model_config: Dict[str, Any]) -> None:
    """
    Validate model configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    if 'type' not in model_config:
        raise ValueError("Model type must be specified")
    
    model_type = model_config['type']
    available_models = get_available_models()
    
    if model_type not in available_models:
        raise ValueError(f"Invalid model type. Available types: {list(available_models.keys())}")
    
    if model_type == 'transfer':
        if 'model_name' in model_config and model_config['model_name'] not in available_models['transfer']:
            raise ValueError(f"Invalid model name for transfer learning. Available models: {available_models['transfer']}")