#!/usr/bin/env python3
"""
Quick start script - simple few-shot learning in one file
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import argparse

class QuickFewShot:
    """Simple few-shot learning - all in one"""
    
    def __init__(self, data_dir, shots=5):
        self.data_dir = Path(data_dir)
        self.shots = shots
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def prepare_data(self):
        """Load data"""
        self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        
        images, labels = [], []
        for idx, class_name in enumerate(self.class_names):
            class_images = list((self.data_dir / class_name).glob('*.jpg'))[:self.shots]
            images.extend([str(img) for img in class_images])
            labels.extend([idx] * len(class_images))
        
        return images, labels
    
    def create_model(self, num_classes):
        """Create simple model"""
        backbone = models.resnet18(pretrained=True)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Linear(num_features, num_classes)
        return backbone.to(self.device)
    
    def train(self, epochs=20):
        """Quick training"""
        # Prepare data
        images, labels = self.prepare_data()
        dataset = SimpleDataset(images, labels, self.transform)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create model
        self.model = self.create_model(len(self.class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train
        print(f"Training on {len(images)} images from {len(self.class_names)} classes...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
        
        print("Training complete!")
        
    def predict(self, image_path):
        """Predict single image"""
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
        
        return self.class_names[pred.item()], conf.item() * 100

class SimpleDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train', help='Training data directory')
    parser.add_argument('--test', type=str, help='Test image path')
    parser.add_argument('--shots', type=int, default=5, help='Shots per class')
    
    args = parser.parse_args()
    
    # Train
    fs = QuickFewShot(args.data, shots=args.shots)
    fs.train(epochs=20)
    
    # Test
    if args.test:
        class_name, confidence = fs.predict(args.test)
        print(f"\nPrediction: {class_name} ({confidence:.1f}% confidence)")