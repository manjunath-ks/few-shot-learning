import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import random
from typing import Tuple, List

class FewShotDataset(Dataset):
    """Dataset for few-shot learning"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label, image_path

def prepare_few_shot_data(data_dir: str, shots_per_class: int = 5, 
                         validation_split: float = 0.2) -> Tuple:
    """
    Prepare data for few-shot learning
    
    Returns:
        train_paths, train_labels, val_paths, val_labels, class_names
    """
    data_path = Path(data_dir)
    
    # Collect all images
    image_paths = []
    labels = []
    class_names = []
    
    # Get all class directories
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_names.append(class_name)
        
        # Get all images in class
        images = list(class_dir.glob('*.jpg')) + \
                list(class_dir.glob('*.jpeg')) + \
                list(class_dir.glob('*.png'))
        
        # Randomly sample shots_per_class images
        if len(images) > shots_per_class:
            images = random.sample(images, shots_per_class)
        
        # Add to lists
        for img_path in images:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    # Create train/val split
    n_samples = len(image_paths)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    split_point = int(n_samples * (1 - validation_split))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    # Split data
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_paths, train_labels, val_paths, val_labels, class_names