"""
Training script for AG News Classifier
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


class Trainer:
    """
    Trainer class for AG News Classifier
    """
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
            accuracy: Training accuracy
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                logits = outputs[1] if isinstance(outputs, tuple) else outputs
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                total_loss += loss.item()
                
                # Calculate accuracy
                logits = outputs[1] if isinstance(outputs, tuple) else outputs
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs, save_path=None):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Optional path to save the best model
            
        Returns:
            history: Training history
        """
        best_val_acc = 0
        
        print(f"\n{'='*50}")
        print(f"Starting Training for {num_epochs} Epochs")
        print(f"{'='*50}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Validation samples: {len(self.val_loader.dataset):,}")
        print(f"{'='*50}\n")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f"  Best model saved (Val Acc: {val_acc:.4f})")
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Best Val Acc: {best_val_acc:.4f}")
        
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"{'='*50}\n")
        
        return self.history
