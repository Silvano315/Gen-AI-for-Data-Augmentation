import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseModel

class BaselineCNN(BaseModel):
    def __init__(self, num_classes: int, input_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_2(x)
        x = self.fc3(x)
        return x

class TransferLearningModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, use_custom_classifier: bool = False):
        super(TransferLearningModel, self).__init__()
        
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        self.use_custom_classifier = use_custom_classifier
        
        if use_custom_classifier:
            self.base_model.reset_classifier(0)
            
            with torch.no_grad():
                sample_input = torch.randn(1, 3, 224, 224)
                sample_output = self.base_model.forward_features(sample_input)
                num_ftrs = sample_output.reshape(sample_output.size(0), -1).size(1)
            
            self.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(256),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = self.base_model.get_classifier()

    def forward(self, x):
        features = self.base_model.forward_features(x)
        if self.use_custom_classifier:
            features = features.reshape(features.size(0), -1)
            return self.classifier(features)
        else:
            return self.base_model.forward_head(features)
    
    def freeze_layers(model: nn.Module, num_layers: int = -1):
        """
        Freeze layers of the model for transfer learning.
        
        Args:
        model (nn.Module): The model to freeze layers in.
        num_layers (int): Number of layers to freeze from the start. -1 means freeze all except the classifier.
        """
        if isinstance(model, TransferLearningModel):
            if num_layers == -1:
                for name, param in model.base_model.named_parameters():
                    if "classifier" not in name and "fc" not in name:
                        param.requires_grad = False
            else:
                for i, (name, param) in enumerate(model.base_model.named_parameters()):
                    if i < num_layers:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            
            if model.use_custom_classifier:
                for param in model.classifier.parameters():
                    param.requires_grad = True
            else:
                for param in model.base_model.get_classifier().parameters():
                    param.requires_grad = True
        else:
            raise NotImplementedError("Freezing layers is only implemented for TransferLearningModel")