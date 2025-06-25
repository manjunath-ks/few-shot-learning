import torch
import torch.nn.functional as F
from PIL import Image
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

class FewShotPredictor:
    """Predictor for few-shot learning models"""
    
    def __init__(self, model_path: str, class_names: List[str], 
                 transform, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.transform = transform
        
        # Load model
        from .model import FewShotModel
        self.model = FewShotModel(len(class_names))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict_single(self, image_path: str, top_k: int = 3) -> List[Dict]:
        """Predict single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top k predictions
        top_probs, top_indices = probabilities.topk(top_k)
        
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item() * 100
            results.append({
                'class': self.class_names[class_idx],
                'confidence': confidence,
                'index': class_idx
            })
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict multiple images"""
        results = []
        
        for image_path in image_paths:
            predictions = self.predict_single(image_path)
            results.append({
                'image': image_path,
                'predictions': predictions
            })
        
        return results
    
    def visualize_prediction(self, image_path: str, save_path: str = None):
        """Visualize prediction with confidence bars"""
        import matplotlib.pyplot as plt
        
        # Get predictions
        predictions = self.predict_single(image_path, top_k=5)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        image = Image.open(image_path)
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Input Image')
        
        # Show predictions
        classes = [p['class'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        y_pos = np.arange(len(classes))
        ax2.barh(y_pos, confidences)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes)
        ax2.invert_yaxis()
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title('Top 5 Predictions')
        ax2.set_xlim(0, 100)
        
        # Add confidence values on bars
        for i, (c, conf) in enumerate(zip(classes, confidences)):
            ax2.text(conf + 1, i, f'{conf:.1f}%', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        return predictions