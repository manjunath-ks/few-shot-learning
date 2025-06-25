import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
for dir_path in [MODEL_DIR, OUTPUT_DIR, OUTPUT_DIR / "predictions", 
                  OUTPUT_DIR / "confusion_matrices", OUTPUT_DIR / "training_plots"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'model_name': 'resnet50',  # Options: resnet18, resnet50, efficientnet_b0
    'freeze_backbone': True,
    'dropout_rate': 0.5,
    'hidden_dims': [512, 256]
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.001,
    'shots_per_class': 5,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5
}

# Data configuration
DATA_CONFIG = {
    'image_size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    'random_crop_scale': (0.8, 1.0),
    'rotation_degrees': 15,
    'color_jitter': {
        'brightness': 0.4,
        'contrast': 0.4,
        'saturation': 0.4,
        'hue': 0.1
    }
}