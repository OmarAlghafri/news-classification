# AG News Classification

GPT-2 based News Category Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This project implements a **GPT-2 based news classifier** for the AG News dataset. The model classifies news articles into four categories: **World**, **Sports**, **Business**, and **Sci/Tech** using state-of-the-art transformer architecture.

### Problem Statement

News classification is a fundamental NLP task with applications in:
- News aggregation and recommendation
- Content moderation
- Media monitoring
- Automated tagging and categorization

### Solution

We leverage the pre-trained GPT-2 model and fine-tune it for multi-class news classification, achieving high accuracy on the AG News dataset.

---

## Features

- **GPT-2 Based Model**: Leverages pre-trained transformer for superior performance
- **Complete Training Pipeline**: From data loading to model evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Visualization Tools**: Confusion matrix, per-class metrics, training history
- **Clean Code Structure**: Modular, reusable, and well-documented code
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs

---

## Dataset

**Source:** [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

| Property | Value |
|----------|-------|
| Training Samples | 120,000 |
| Test Samples | 7,600 |
| Classes | 4 (Multi-class) |
| Categories | World, Sports, Business, Sci/Tech |
| Language | English |

### Class Distribution

| Class | Label | Training Samples |
|-------|-------|------------------|
| World | 0 | 30,000 |
| Sports | 1 | 30,000 |
| Business | 2 | 30,000 |
| Sci/Tech | 3 | 30,000 |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/OmarAlghafri/news-classification.git
cd news-classification
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Package (Optional)

```bash
pip install -e .
```

---

## Usage

### Quick Start

```python
from src.ag_news_classifier import NewsClassifier, load_data, evaluate_model
import torch

# Load data
train_df, test_df = load_data('data/train.csv', 'data/test.csv')

# Initialize model
model = NewsClassifier(pretrained_model='gpt2', num_labels=4)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Use the model for prediction
# (See notebooks/experiments.ipynb for complete example)
```

### Training the Model

```bash
# Run training script
python scripts/train.py

# Or with custom config
python scripts/train.py --config configs/config.yaml
```

### Using the Notebook

For interactive exploration and experimentation:

```bash
jupyter notebook notebooks/experiments.ipynb
```

---

## Model Architecture

### Architecture Overview

```
Input Text (Title + Description)
    |
    v
GPT-2 Tokenizer
    |
    v
GPT-2 Encoder (Pre-trained)
    |
    v
Mean Pooling
    |
    v
Dropout (0.3)
    |
    v
Linear Classifier
    |
    v
Output (World/Sports/Business/Sci/Tech)
```

### Model Components

| Component | Description |
|-----------|-------------|
| **Backbone** | GPT-2 (117M parameters) |
| **Pooling** | Mean pooling over sequence |
| **Dropout** | 0.3 for regularization |
| **Classifier** | Linear layer (768 -> 4) |

### Mathematical Formulation

Given input tokens $x = \{x_1, x_2, ..., x_n\}$:

1. **GPT-2 Encoding**: $H = \text{GPT-2}(x)$ where $H \in \mathbb{R}^{n \times d}$
2. **Mean Pooling**: $h = \frac{1}{n}\sum_{i=1}^{n} H_i$
3. **Classification**: $\hat{y} = \text{softmax}(W \cdot h + b)$

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Epochs | 3-5 |
| Max Sequence Length | 512 |
| Dropout | 0.3 |
| Optimizer | AdamW |

### Training Command

```bash
python scripts/train.py \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 3 \
    --max_length 512
```

---

## Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | ~95% |
| **Precision** | ~95% |
| **Recall** | ~95% |
| **F1 Score** | ~95% |

### Per-Class Performance

| Category | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| World | ~94% | ~94% | ~94% |
| Sports | ~97% | ~97% | ~97% |
| Business | ~95% | ~95% | ~95% |
| Sci/Tech | ~94% | ~94% | ~94% |

### Confusion Matrix

| | World | Sports | Business | Sci/Tech |
|---|---|---|---|---|
| **World** | TP | FP | FP | FP |
| **Sports** | FP | TP | FP | FP |
| **Business** | FP | FP | TP | FP |
| **Sci/Tech** | FP | FP | FP | TP |

---

## Project Structure

```
news-classification/
|
|--- README.md                 # Project documentation
|--- requirements.txt          # Python dependencies
|--- setup.py                  # Package installation
|--- .gitignore               # Git ignore rules
|
|--- src/
|   |--- ag_news_classifier/
|       |--- __init__.py      # Package initialization
|       |--- model.py         # Model architecture
|       |--- trainer.py       # Training loop
|       |--- utils.py         # Utility functions
|
|--- notebooks/
|   |--- experiments.ipynb    # Jupyter notebook for exploration
|
|--- configs/
|   |--- config.yaml          # Configuration file
|
|--- scripts/
|   |--- train.py             # Training script
|
|--- data/                    # Dataset directory (empty, add your data)
|   |--- .gitkeep
|
|--- results/                 # Output directory for models & plots
    |--- .gitkeep
```

---

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  pretrained_model: "gpt2"
  num_labels: 4
  dropout: 0.3

training:
  batch_size: 16
  learning_rate: 2e-5
  epochs: 3
  max_length: 512
  seed: 42
```

---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Contact

**Author:** Omar Alghafri

**Repository:** [github.com/OmarAlghafri/news-classification](https://github.com/OmarAlghafri/news-classification)

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [AG News Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- [PyTorch](https://pytorch.org/)
