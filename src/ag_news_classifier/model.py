"""
Model architecture for AG News Classification
Using GPT-2 based model for news category classification
"""

import torch
import torch.nn as nn
from transformers import GPT2Model


class NewsClassifier(nn.Module):
    """
    GPT-2 based News Classifier for multi-class classification (World, Sports, Business, Sci/Tech)
    """
    
    def __init__(self, pretrained_model='gpt2', num_labels=4, dropout=0.3):
        """
        Initialize the news classifier.
        
        Args:
            pretrained_model: Pre-trained GPT-2 model name
            num_labels: Number of classification labels (4 for AG News)
            dropout: Dropout rate for classification head
        """
        super(NewsClassifier, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(pretrained_model)
        
        # Freeze GPT-2 parameters (optional - can be unfrozen for fine-tuning)
        for param in self.gpt2.parameters():
            param.requires_grad = True
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_labels)
        
        # Class labels mapping
        self.label_map = {
            0: 'World',
            1: 'Sports',
            2: 'Business',
            3: 'Sci/Tech'
        }
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            labels: Optional labels for loss computation
            
        Returns:
            logits: Classification logits
            loss: Optional loss if labels are provided
        """
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use the last hidden state
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        # Mean pooling over sequence length
        if attention_mask is not None:
            # Apply attention mask for mean pooling
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.shape)
            sum_hidden_states = (hidden_states * attention_mask_expanded).sum(dim=1)
            mean_hidden_states = sum_hidden_states / attention_mask.sum(dim=1, keepdim=True)
        else:
            mean_hidden_states = hidden_states.mean(dim=1)
        
        # Classification
        logits = self.classifier(self.dropout(mean_hidden_states))
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            
        return loss, logits if loss is not None else logits
    
    def predict_category(self, logits):
        """
        Convert logits to category predictions.
        
        Args:
            logits: Model output logits
            
        Returns:
            predictions: List of predicted category names
        """
        predictions = torch.argmax(logits, dim=1)
        return [self.label_map[p.item()] for p in predictions]


class AGNewsDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for AG News articles
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of news article texts
            labels: List of labels (0-3 for the 4 categories)
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
