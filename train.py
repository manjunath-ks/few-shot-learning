#!/usr/bin/env python3
"""
Main training script for few-shot learning
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.config import *
from src.dataset import FewShotDataset, prepare_few_shot_data
from src.model import FewShotModel
from src.trainer import FewShotTrainer
from src.utils import get_transforms, save_class_names

def main(args):
    # Set random seeds
    torch.manual_seed(42)
    
    # Prepare data
    print("Preparing data...")
    train_paths, train_labels, val_paths, val_labels, class_names = prepare_few_shot_data(
        args.data_dir or str(DATA_DIR / 'train'),
        shots_per_class=args.shots_per_class or TRAIN_CONFIG['shots_per_class'],
        validation_split=TRAIN_CONFIG['validation_split']
    )
    
    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Save class names
    save_class_names(class_names, MODEL_DIR / 'class_names.json')
    
    # Create datasets and loaders
    train_transform, val_transform = get_transforms(DATA_CONFIG)
    
    train_dataset = FewShotDataset(train_paths, train_labels, train_transform)
    val_dataset = FewShotDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size or TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size or TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print(f"\nCreating {MODEL_CONFIG['model_name']} model...")
    model = FewShotModel(
        num_classes=len(class_names),
        **MODEL_CONFIG
    )
    
    # Train
    trainer = FewShotTrainer(model)
    print("\nStarting training...")
    
    best_acc = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs or TRAIN_CONFIG['epochs'],
        lr=args.lr or TRAIN_CONFIG['learning_rate'],
        save_dir=str(MODEL_DIR)
    )
    
    # Plot training history
    trainer.plot_history(str(OUTPUT_DIR / 'training_plots' / 'history.png'))
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {MODEL_DIR / 'best_model.pth'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Few-Shot Learning Model')
    parser.add_argument('--data-dir', type=str, help='Path to training data')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--shots-per-class', type=int, help='Shots per class')
    
    args = parser.parse_args()
    main(args)