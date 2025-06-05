"""
Base classes for NER model implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NERPrediction:
    """
    Standardized NER prediction output.
    
    Attributes:
        start: Character start position
        end: Character end position  
        entity_type: Type of entity (e.g., 'PERSON', 'ORG')
        text: The actual text span
        confidence: Confidence score (optional)
    """
    start: int
    end: int
    entity_type: str
    text: str
    confidence: Optional[float] = None
    
    def to_tuple(self) -> Tuple[int, int, str]:
        """Convert to evaluation-compatible tuple format."""
        return (self.start, self.end, self.entity_type)


class BaseNERModel(ABC):
    """
    Abstract base class for all NER model implementations.
    
    This provides a unified interface for different NER models including
    API-based models (OpenAI, Mistral), local models (spaCy, DeepPavlov),
    and transformer models (HuggingFace).
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the NER model.
        
        Args:
            model_name: Name/identifier of the model
            **kwargs: Model-specific configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model (load weights, authenticate APIs, etc.).
        
        This is separate from __init__ to allow for lazy loading and
        better error handling in Google Colab environments.
        """
        pass
    
    @abstractmethod
    def predict(self, text: str) -> List[NERPrediction]:
        """
        Predict named entities in a single text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of NER predictions with character positions
            
        Raises:
            RuntimeError: If model is not initialized
            ValueError: If text is invalid
        """
        pass
    
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'config': self.config.copy(),
            'is_initialized': self.is_initialized
        }
    
    def cleanup(self) -> None:
        """
        Clean up model resources (memory, connections, etc.).
        
        Default implementation does nothing. Override for models
        that need explicit cleanup.
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', initialized={self.is_initialized})"


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ModelInitializationError(ModelError):
    """Raised when model initialization fails."""
    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class UnsupportedModelError(ModelError):
    """Raised when trying to create an unsupported model type."""
    pass