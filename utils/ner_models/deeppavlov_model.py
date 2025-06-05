"""
DeepPavlov NER model implementation.
"""

import logging
from typing import List, Dict, Any, Optional
from .base import BaseNERModel, NERPrediction, ModelInitializationError, ModelPredictionError

logger = logging.getLogger(__name__)


class DeepPavlovModel(BaseNERModel):
    """
    DeepPavlov NER model implementation.
    
    Supports Russian-specific models like ner_collection3_bert and 
    multi-language models like ner_ontonotes_bert_mult.
    """
    
    # Available DeepPavlov NER models
    AVAILABLE_MODELS = {
        'ner_collection3_bert': 'Russian-specific BERT-based NER model',
        'ner_ontonotes_bert_mult': 'Multi-language BERT-based NER model',
        'ner_ontonotes_bert': 'English BERT-based NER model',
        'ner_rus_bert': 'Russian BERT NER model (alternative)'
    }
    
    def __init__(self, model_name: str = "ner_collection3_bert", 
                 entity_types: Optional[set] = None, **kwargs):
        """
        Initialize DeepPavlov NER model.
        
        Args:
            model_name: DeepPavlov model identifier
            entity_types: Set of entity types to filter (e.g., {'PERSON', 'PER'})
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        
        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Model '{model_name}' not in known models. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.entity_types = entity_types or {'PERSON', 'PER'}
        self.model = None
        self._installation_attempted = False
        
    def initialize(self) -> None:
        """
        Initialize the DeepPavlov model.
        
        Downloads and builds the model if necessary.
        """
        if self.is_initialized:
            return
            
        try:
            # Import DeepPavlov
            logger.info(f"Importing DeepPavlov for model {self.model_name}")
            from deeppavlov import build_model
            
            # Build and download model
            logger.info(f"Building DeepPavlov model: {self.model_name}")
            self.model = build_model(
                self.model_name, 
                download=True, 
                install=not self._installation_attempted
            )
            self._installation_attempted = True
            
            # Test the model with a simple example
            test_result = self.model(["Test"])
            if not test_result or len(test_result) < 2:
                raise ModelInitializationError(f"Model {self.model_name} returned unexpected output format")
            
            self.is_initialized = True
            logger.info(f"Successfully initialized DeepPavlov model: {self.model_name}")
            
        except ImportError as e:
            raise ModelInitializationError(
                f"DeepPavlov not installed. Install with: pip install deeppavlov. Error: {e}"
            )
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize DeepPavlov model {self.model_name}: {e}")
    
    def predict(self, text: str) -> List[NERPrediction]:
        """
        Predict named entities in text using DeepPavlov model.
        
        Args:
            text: Input text to process
            
        Returns:
            List of NER predictions
        """
        if not self.is_initialized:
            raise RuntimeError(f"Model {self.model_name} not initialized. Call initialize() first.")
        
        if not text or not text.strip():
            return []
        
        try:
            # DeepPavlov expects text wrapped in a list
            result = self.model([text])
            
            # Validate result format
            if not result or len(result) < 2:
                logger.warning(f"DeepPavlov model returned unexpected format for text: {text[:50]}...")
                return []
            
            tokens = result[0][0] if result[0] else []
            labels = result[1][0] if result[1] else []
            
            if len(tokens) != len(labels):
                logger.warning(f"Token/label mismatch: {len(tokens)} tokens, {len(labels)} labels")
                return []
            
            # Convert to NER predictions
            predictions = self._convert_to_predictions(text, tokens, labels)
            
            # Filter by entity types if specified
            if self.entity_types:
                predictions = [p for p in predictions if p.entity_type in self.entity_types]
            
            logger.debug(f"DeepPavlov model {self.model_name} found {len(predictions)} entities")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with DeepPavlov model {self.model_name}: {e}")
            raise ModelPredictionError(f"DeepPavlov prediction failed: {e}")
    
    def _convert_to_predictions(self, text: str, tokens: List[str], labels: List[str]) -> List[NERPrediction]:
        """
        Convert DeepPavlov tokens and labels to NER predictions.
        
        Args:
            text: Original text
            tokens: List of tokens from DeepPavlov
            labels: Corresponding BIO labels
            
        Returns:
            List of NER predictions with character positions
        """
        predictions = []
        current_entity = None
        start_pos = None
        entity_tokens = []
        
        # Calculate token positions in original text
        token_positions = self._calculate_token_positions(text, tokens)
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                # Begin new entity - save previous if exists
                if current_entity is not None and start_pos is not None:
                    end_pos = token_positions[i] if i < len(token_positions) else len(text)
                    entity_text = text[start_pos:end_pos].strip()
                    if entity_text:
                        predictions.append(NERPrediction(
                            start=start_pos,
                            end=end_pos,
                            entity_type=current_entity,
                            text=entity_text,
                            confidence=None
                        ))
                
                # Start new entity
                current_entity = label.split('-', 1)[1]  # Remove B- prefix
                start_pos = token_positions[i] if i < len(token_positions) else None
                entity_tokens = [token]
                
            elif label.startswith('I-') and current_entity is not None:
                # Continue current entity
                entity_tokens.append(token)
                
            elif label.startswith('E-') and current_entity is not None:
                # End current entity (some models use E- tag)
                entity_tokens.append(token)
                end_pos = (token_positions[i] + len(token)) if i < len(token_positions) else len(text)
                entity_text = text[start_pos:end_pos].strip()
                if entity_text:
                    predictions.append(NERPrediction(
                        start=start_pos,
                        end=end_pos,
                        entity_type=current_entity,
                        text=entity_text,
                        confidence=None
                    ))
                current_entity = None
                start_pos = None
                entity_tokens = []
                
            else:
                # Outside entity or different entity - save previous if exists
                if current_entity is not None and start_pos is not None:
                    end_pos = token_positions[i] if i < len(token_positions) else len(text)
                    entity_text = text[start_pos:end_pos].strip()
                    if entity_text:
                        predictions.append(NERPrediction(
                            start=start_pos,
                            end=end_pos,
                            entity_type=current_entity,
                            text=entity_text,
                            confidence=None
                        ))
                current_entity = None
                start_pos = None
                entity_tokens = []
        
        # Handle entity that continues to end of text
        if current_entity is not None and start_pos is not None:
            entity_text = text[start_pos:].strip()
            if entity_text:
                predictions.append(NERPrediction(
                    start=start_pos,
                    end=len(text),
                    entity_type=current_entity,
                    text=entity_text,
                    confidence=None
                ))
        
        return predictions
    
    def _calculate_token_positions(self, text: str, tokens: List[str]) -> List[int]:
        """
        Calculate character positions of tokens in original text.
        
        Args:
            text: Original text
            tokens: List of tokens
            
        Returns:
            List of start positions for each token
        """
        positions = []
        current_pos = 0
        
        for token in tokens:
            # Find token in text starting from current position
            token_start = text.find(token, current_pos)
            if token_start == -1:
                # Token not found - try approximate position
                logger.warning(f"Token '{token}' not found in text at position {current_pos}")
                positions.append(current_pos)
                current_pos += len(token)  # Approximate
            else:
                positions.append(token_start)
                current_pos = token_start + len(token)
        
        return positions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the DeepPavlov model."""
        info = super().get_model_info()
        info.update({
            'provider': 'DeepPavlov',
            'model_description': self.AVAILABLE_MODELS.get(self.model_name, 'Unknown DeepPavlov model'),
            'entity_types_filter': list(self.entity_types) if self.entity_types else None,
            'available_models': list(self.AVAILABLE_MODELS.keys())
        })
        return info
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            # DeepPavlov models don't need explicit cleanup
            self.model = None
            self.is_initialized = False
            logger.info(f"Cleaned up DeepPavlov model: {self.model_name}")


# Convenience functions for creating DeepPavlov models
def create_russian_ner_model(**kwargs) -> DeepPavlovModel:
    """
    Create Russian-specific DeepPavlov NER model.
    
    Returns:
        Configured DeepPavlov model for Russian NER
    """
    return DeepPavlovModel(
        model_name="ner_collection3_bert",
        entity_types={'PERSON', 'PER'},
        **kwargs
    )


def create_multilang_ner_model(**kwargs) -> DeepPavlovModel:
    """
    Create multi-language DeepPavlov NER model.
    
    Returns:
        Configured DeepPavlov model for multi-language NER
    """
    return DeepPavlovModel(
        model_name="ner_ontonotes_bert_mult",
        entity_types={'PERSON', 'PER'},
        **kwargs
    )