"""
Utility functions for AG News Classification
"""

import os
import re
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(train_path, test_path=None):
    """
    Load AG News dataset from CSV file(s).
    
    Args:
        train_path: Path to the training CSV file
        test_path: Optional path to the test CSV file
        
    Returns:
        train_df: pandas DataFrame with training data
        test_df: pandas DataFrame with test data (if provided)
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Data file not found: {train_path}")
    
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df):,} training samples")
    
    test_df = None
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"Loaded {len(test_df):,} test samples")
    
    return train_df, test_df


def preprocess_text(text):
    """
    Basic text preprocessing for news articles.
    
    Args:
        text: Raw article text
        
    Returns:
        Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s.!?]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def prepare_data(df, text_columns=['Title', 'Description']):
    """
    Prepare data for training by combining text columns and adjusting labels.
    
    Args:
        df: DataFrame with news data
        text_columns: List of columns to combine for text
        
    Returns:
        df: DataFrame with prepared data
    """
    # Combine text columns
    df['text'] = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    
    # Adjust labels to 0-indexed (AG News has 1-4, we need 0-3)
    if 'Class Index' in df.columns:
        df['label'] = df['Class Index'] - 1
    
    return df


def evaluate_model(model, dataloader, device, num_classes=4):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[1] if isinstance(outputs, tuple) else outputs
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': classification_report(all_labels, all_preds, target_names=class_names, zero_division=0),
        'predictions': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }
    
    # Calculate per-class metrics
    metrics['precision_per_class'] = precision_score(all_labels, all_preds, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(all_labels, all_preds, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    return metrics


def plot_metrics(metrics, save_path=None):
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from evaluate_model
        save_path: Optional path to save the figure
    """
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].tick_params(axis='y', rotation=0)
    
    # Accuracy bar chart
    acc_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    acc_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    axes[0, 1].bar(acc_metrics, acc_values, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_title('Overall Performance Metrics')
    axes[0, 1].set_ylabel('Score')
    for i, v in enumerate(acc_values):
        axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # Per-class F1 scores
    axes[1, 0].bar(class_names, metrics['f1_per_class'], color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].set_title('F1 Score by Category')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(metrics['f1_per_class']):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # Prediction distribution
    axes[1, 1].hist(metrics['probs'], bins=50, alpha=0.7)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Predicted Probabilities')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to: {save_path}")
    
    plt.show()


def print_metrics(metrics):
    """
    Print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from evaluate_model
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(metrics['classification_report'])
    print("="*50)
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("="*50)
