#!/usr/bin/env python3
"""
Training Script for AG News Classifier

Usage:
    python scripts/train.py [--config CONFIG] [--data_path PATH] [--epochs N]

Example:
    python scripts/train.py --config configs/config.yaml --epochs 3
"""

import os
import sys
import argparse
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch.utils.data import DataLoader, random_split
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from ag_news_classifier.model import NewsClassifier, AGNewsDataset
from ag_news_classifier.trainer import Trainer
from ag_news_classifier.utils import (
    load_data, preprocess_text, prepare_data,
    evaluate_model, plot_metrics, print_metrics
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AG News Classifier')
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--train_path', type=str, default=None,
                        help='Path to training dataset CSV file')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to test dataset CSV file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    
    # Set defaults from config or command line
    pretrained_model = model_config.get('pretrained_model', 'gpt2')
    num_labels = model_config.get('num_labels', 4)
    dropout = model_config.get('dropout', 0.3)
    
    batch_size = args.batch_size or training_config.get('batch_size', 16)
    learning_rate = args.learning_rate or training_config.get('learning_rate', 2e-5)
    epochs = args.epochs or training_config.get('epochs', 3)
    max_length = args.max_length or training_config.get('max_length', 512)
    seed = args.seed or training_config.get('seed', 42)
    
    train_path = args.train_path or data_config.get('train_path', 
                  '/kaggle/input/ag-news-classification-dataset/train.csv')
    test_path = args.test_path or data_config.get('test_path',
                  '/kaggle/input/ag-news-classification-dataset/test.csv')
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print("AG News Classification - Training")
    print(f"{'='*50}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}\n")
    
    # Load and preprocess data
    print("Loading Data...")
    
    # Try to load data
    if not os.path.exists(train_path):
        # Try alternative paths
        alt_paths = [
            'data/train.csv',
            '../data/train.csv',
            '/kaggle/input/ag-news-classification-dataset/train.csv'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                train_path = alt_path
                break
    
    if not os.path.exists(train_path):
        print(f"Error: Data file not found at {train_path}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset")
        return
    
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df):,} training samples")
    
    # Preprocess text
    print("Preprocessing Data...")
    train_df = prepare_data(train_df)
    train_df['text'] = train_df['text'].apply(preprocess_text)
    
    # Split data for validation
    print("Splitting Data...")
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=seed, stratify=train_df['label'])
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    
    # Initialize tokenizer
    print("\nInitializing Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # Create datasets
    print("Creating Datasets...")
    train_dataset = AGNewsDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length=max_length
    )
    val_dataset = AGNewsDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\nInitializing Model...")
    model = NewsClassifier(
        pretrained_model=pretrained_model,
        num_labels=num_labels,
        dropout=dropout
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=num_training_steps
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # Train
    save_path = os.path.join(output_dir, 'best_model.pt')
    history = trainer.train(num_epochs=epochs, save_path=save_path)
    
    # Evaluate on validation set
    print("\nEvaluating on Validation Set...")
    metrics = evaluate_model(model, val_loader, device)
    print_metrics(metrics)
    
    # Plot metrics
    plot_path = os.path.join(output_dir, 'metrics.png')
    plot_metrics(metrics, save_path=plot_path)
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    history_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nTraining history saved to: {history_path}")
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    print(f"Model saved to: {save_path}")
    print(f"Metrics plot saved to: {plot_path}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
