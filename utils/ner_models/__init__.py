"""
Simplified NER Models Package

This package provides a unified interface for various Named Entity Recognition models
including OpenAI API models, spaCy models, and DeepPavlov models.

Simple, direct model instantiation without factory complexity.
"""

from .base import BaseNERModel, NERPrediction
from .openai_model import OpenAIModel
from .spacy_model import SpacyModel
from .deeppavlov_model import DeepPavlovModel

__all__ = [
    'BaseNERModel',
    'NERPrediction',
    'OpenAIModel',
    'SpacyModel',
    'DeepPavlovModel'
]

