import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class FewShotModel(nn.Module):
    """Few-shot learning model using transfer learning"""
    
    def __init__(self, num_classes: int, model_name: str = 'resnet50', 
                 freeze_backbone: bool = True, dropout_rate: float = 0.5,
                 hidden_dims: list = [512, 256]):
        super(FewShotModel, self).__init__()
        
        # Load pre-trained backbone
        self.backbone, num_features = self._get_backbone(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Unfreeze last few layers
            self._unfreeze_layers(model_name)
        
        # Create classifier head
        layers = []
        input_dim = num_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
    def _get_backbone(self, model_name: str):
        """Get pre-trained backbone"""
        if model_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            num_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif model_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            num_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif model_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=True)
            num_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return backbone, num_features
    
    def _unfreeze_layers(self, model_name: str):
        """Unfreeze last few layers for fine-tuning"""
        if 'resnet' in model_name:
            # Unfreeze layer4
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """Extract features without classification"""
        with torch.no_grad():
            return self.backbone(x)