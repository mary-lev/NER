"""
spaCy-based NER model implementation.
"""

import logging
from typing import List, Dict, Any, Set

from .base import BaseNERModel, NERPrediction, ModelInitializationError, ModelPredictionError

logger = logging.getLogger(__name__)


class SpacyModel(BaseNERModel):
    """
    spaCy-based NER model with support for different models and entity types.
    
    Supports various spaCy models including custom Russian models that can be
    installed from HuggingFace or other sources.
    """
    
    # Default entity types to extract (can be customized)
    DEFAULT_PERSON_LABELS = {'PERSON', 'PER'}
    
    def __init__(self, model_name: str = "ru_spacy_ru_updated", entity_types: Set[str] = None, **kwargs):
        """
        Initialize spaCy model.
        
        Args:
            model_name: spaCy model name (e.g., "en_core_web_sm", "ru_spacy_ru_updated")
            entity_types: Set of entity types to extract (default: person entities)
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        self.entity_types = entity_types or self.DEFAULT_PERSON_LABELS
        self.nlp = None
        
        # Configuration options
        self.max_length = kwargs.get('max_length', 1000000)  # spaCy default limit
        self.disable_components = kwargs.get('disable', [])  # Components to disable for speed
        
    def initialize(self) -> None:
        """Initialize spaCy model and load pipeline."""
        try:
            import spacy
            
            # Try to load the model
            try:
                self.nlp = spacy.load(
                    self.model_name,
                    disable=self.disable_components
                )
                
                # Set max length if specified
                if self.max_length != 1000000:
                    self.nlp.max_length = self.max_length
                
                logger.info(f"spaCy model {self.model_name} loaded successfully")
                
            except OSError:
                # Model not found, try to provide helpful installation instructions
                installation_commands = self._get_installation_commands()
                raise ModelInitializationError(
                    f"spaCy model '{self.model_name}' not found.\n"
                    f"Try installing with one of these commands:\n"
                    f"{installation_commands}"
                )
            
            # Verify the model has NER capability
            if 'ner' not in self.nlp.pipe_names:
                logger.warning(f"Model {self.model_name} does not have NER component")
            
            self.is_initialized = True
            
        except ImportError:
            raise ModelInitializationError(
                "spaCy not available. Install with: !pip install spacy"
            )
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize spaCy model: {e}")
    
    def _get_installation_commands(self) -> str:
        """Get installation commands for common spaCy models."""
        commands = {
            'en_core_web_sm': '!python -m spacy download en_core_web_sm',
            'en_core_web_md': '!python -m spacy download en_core_web_md',
            'en_core_web_lg': '!python -m spacy download en_core_web_lg',
            'ru_core_news_sm': '!python -m spacy download ru_core_news_sm',
            'ru_core_news_md': '!python -m spacy download ru_core_news_md',
            'ru_core_news_lg': '!python -m spacy download ru_core_news_lg',
            'ru_spacy_ru_updated': '!pip install https://huggingface.co/Dessan/ru_spacy_ru_updated/resolve/main/ru_spacy_ru_updated-any-py3-none-any.whl'
        }
        
        specific_command = commands.get(self.model_name)
        if specific_command:
            return specific_command
        else:
            return (
                f"!python -m spacy download {self.model_name}\n"
                f"# or if it's a custom model:\n"
                f"!pip install {self.model_name}"
            )
    
    def predict(self, text: str) -> List[NERPrediction]:
        """
        Predict named entities using spaCy.
        
        Args:
            text: Input text to process
            
        Returns:
            List of NER predictions
        """
        if not self.is_initialized:
            raise RuntimeError(f"Model {self.model_name} not initialized")
        
        if not text or not text.strip():
            return []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            predictions = []
            for ent in doc.ents:
                # Filter by entity type
                if ent.label_ in self.entity_types:
                    predictions.append(NERPrediction(
                        start=ent.start_char,
                        end=ent.end_char,
                        entity_type=self._normalize_entity_type(ent.label_),
                        text=ent.text,
                        confidence=None  # spaCy doesn't provide confidence by default
                    ))
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Error in spaCy prediction for {self.model_name}: {e}")
            raise ModelPredictionError(f"spaCy prediction failed: {e}")
    
    def _normalize_entity_type(self, spacy_label: str) -> str:
        """
        Normalize spaCy entity labels to consistent format.
        
        Args:
            spacy_label: Original spaCy entity label
            
        Returns:
            Normalized entity type
        """
        # Map common spaCy labels to standard format
        label_mapping = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'PERS': 'PERSON',
            'ORG': 'ORG',
            'ORGANIZATION': 'ORG',
            'LOC': 'LOC',
            'LOCATION': 'LOC',
            'GPE': 'GPE',
            'MISC': 'MISC'
        }
        
        return label_mapping.get(spacy_label.upper(), spacy_label)
    
    
    def get_available_entity_types(self) -> Set[str]:
        """
        Get all entity types available in the loaded model.
        
        Returns:
            Set of available entity type labels
        """
        if not self.is_initialized:
            raise RuntimeError(f"Model {self.model_name} not initialized")
        
        if 'ner' in self.nlp.pipe_names:
            ner = self.nlp.get_pipe('ner')
            return set(ner.labels)
        else:
            return set()
    
    def update_entity_types(self, entity_types: Set[str]) -> None:
        """
        Update the entity types to extract.
        
        Args:
            entity_types: New set of entity types to extract
        """
        self.entity_types = entity_types
        logger.info(f"Updated entity types for {self.model_name}: {entity_types}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including spaCy-specific details."""
        info = super().get_model_info()
        info.update({
            'provider': 'spaCy',
            'entity_types': list(self.entity_types),
            'max_length': self.max_length,
            'disabled_components': self.disable_components
        })
        
        if self.is_initialized:
            info.update({
                'pipeline_components': self.nlp.pipe_names,
                'available_entity_types': list(self.get_available_entity_types()) if 'ner' in self.nlp.pipe_names else []
            })
        
        return info
    
    def cleanup(self) -> None:
        """Clean up spaCy model resources."""
        if self.nlp is not None:
            # spaCy doesn't require explicit cleanup, but we can clear the reference
            self.nlp = None
            logger.debug(f"Cleaned up spaCy model {self.model_name}")
        
        self.is_initialized = False