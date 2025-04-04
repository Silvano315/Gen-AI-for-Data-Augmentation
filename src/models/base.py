from typing import Dict, Any
import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all models."""
    
    def __init__(self) -> None:
        super().__init__()
        
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            'name': self.__class__.__name__,
            'params': self.state_dict()
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseModel':
        """Create model from configuration."""
        model = cls()
        model.load_state_dict(config['params'])
        return model

    def freeze_layers(self, num_layers: int = -1) -> None:
        """Freeze model layers for transfer learning."""
        raise NotImplementedError