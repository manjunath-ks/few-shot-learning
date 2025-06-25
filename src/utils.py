import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List
import json

def get_transforms(config: dict):
    """Get data transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config['image_size'], 
            scale=config.get('random_crop_scale', (0.8, 1.0))
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(config.get('rotation_degrees', 15)),
        transforms.ColorJitter(**config.get('color_jitter', {})),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(config['image_size'] * 1.14)),
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])
    
    return train_transform, val_transform

def save_class_names(class_names: List[str], path: str):
    """Save class names to JSON file"""
    with open(path, 'w') as f:
        json.dump(class_names, f, indent=2)

def load_class_names(path: str) -> List[str]:
    """Load class names from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return cm

def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 digits=3)
    print("\nClassification Report:")
    print("=" * 50)
    print(report)