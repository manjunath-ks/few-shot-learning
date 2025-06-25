#!/usr/bin/env python3
"""
Main prediction script for few-shot learning
"""
import argparse
from pathlib import Path
import json

from src.config import *
from src.predictor import FewShotPredictor
from src.utils import get_transforms, load_class_names

def main(args):
    # Load class names
    class_names = load_class_names(MODEL_DIR / 'class_names.json')
    
    # Get transform
    _, val_transform = get_transforms(DATA_CONFIG)
    
    # Create predictor
    model_path = args.model_path or str(MODEL_DIR / 'best_model.pth')
    predictor = FewShotPredictor(model_path, class_names, val_transform)
    
    if args.visualize:
        # Visualize prediction
        predictions = predictor.visualize_prediction(
            args.image,
            save_path=str(OUTPUT_DIR / 'predictions' / 'prediction.png')
        )
    else:
        # Text predictions
        predictions = predictor.predict_single(args.image, top_k=args.top_k)
        
        print(f"\nPredictions for {args.image}:")
        print("=" * 50)
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['class']:<20} {pred['confidence']:>6.2f}%")
    
    # Save predictions if requested
    if args.save_json:
        output_path = OUTPUT_DIR / 'predictions' / 'predictions.json'
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\nPredictions saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using Few-Shot Model')
    parser.add_argument('image', type=str, help='Path to image')
    parser.add_argument('--model-path', type=str, help='Path to model')
    parser.add_argument('--top-k', type=int, default=3, help='Top K predictions')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--save-json', action='store_true', help='Save predictions to JSON')
    
    args = parser.parse_args()
    main(args)