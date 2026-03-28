"""
AG News Classifier Package
GPT-2 based News Category Classification
"""

__version__ = "1.0.0"
__author__ = "Omar Alghafri"

from .model import NewsClassifier
from .utils import load_data, preprocess_text, evaluate_model

__all__ = ["NewsClassifier", "load_data", "preprocess_text", "evaluate_model"]
