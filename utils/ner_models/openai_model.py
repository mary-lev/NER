"""
OpenAI API-based NER model implementation.
"""

import os
import re
import ast
import json
import logging
from typing import List, Dict, Any, Optional
import time

from .base import BaseNERModel, NERPrediction, ModelInitializationError, ModelPredictionError

logger = logging.getLogger(__name__)


class OpenAIModel(BaseNERModel):
    """
    OpenAI API-based NER model with configurable model names.
    
    Supports GPT-3.5, GPT-4, GPT-4o, and other OpenAI models for
    Named Entity Recognition with prompt engineering.
    """
    
    # Default prompts for different output formats
    DEFAULT_PROMPTS = {
        'list_format': """You are a Named Entity Recognition (NER) system designed to process Russian text. Your task is to extract the names of real people mentioned in the text. Follow these steps:

1. Read the text and identify the names of the people mentioned in the text.
2. Consider only the names of people if they describe a person. We don't need titles of works of art or names of organizations if they contain the names of people.
3. Return the list of these names in the same form as they appear in the text. Don't change the form of any name!
4. Be sure that all occurences of the names in the text are included in the list even if they are mentioned several times.
Return only the list of names without any additional comments or formatting.

Example 1:
Input: "Встреча с писательницей Сюзанной Кулешовой Презентация книги «Последний глоток божоле на двоих». Кулешова Сюзанна Марковна, член Союза писателей Санкт-Петербурга."
Output: ["Сюзанной Кулешовой", "Кулешова Сюзанна Марковна"]

Example 2:
Input: "Фестиваль Поэзия и вино. 20:00 - о Любви. Милена Райт и Костя Гафнер 21:10 - Открытый Микрофон"
Output: ["Милена Райт", "Костя Гафнер"]""",

        'json_format': """You are a Named Entity Recognition system. Extract person names from Russian text and return them as a JSON list with their character positions.

Return format: [{"name": "extracted_name", "start": start_pos, "end": end_pos}, ...]

Rules:
1. Only extract real person names
2. Include all occurrences even if repeated
3. Preserve exact text form
4. Provide accurate character positions

Example:
Input: "Встреча с Анной Петровой и Иваном Сидоровым"
Output: [{"name": "Анной Петровой", "start": 11, "end": 24}, {"name": "Иваном Сидоровым", "start": 27, "end": 43}]"""
    }
    
    def __init__(self, model_name: str = "gpt-4o", output_format: str = "list", **kwargs):
        """
        Initialize OpenAI model.
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4", "gpt-3.5-turbo")
            output_format: "list" for name lists or "json" for structured output
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        super().__init__(model_name, **kwargs)
        
        self.output_format = output_format
        self.temperature = kwargs.get('temperature', 1.0)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.prompt = kwargs.get('prompt', self.DEFAULT_PROMPTS.get(f'{output_format}_format'))
        
        # API configuration
        self.api_key = kwargs.get('api_key')
        self.client = None
        
        # Rate limiting
        self.requests_per_minute = kwargs.get('requests_per_minute', 60)
        self.last_request_time = 0
        
    def initialize(self) -> None:
        """Initialize OpenAI client and authenticate."""
        try:
            # Try to import OpenAI (will install in Colab if needed)
            import openai
            
            # Get API key
            if not self.api_key:
                # Try environment variable first
                self.api_key = os.getenv('OPENAI_API_KEY')
                
                # In Colab, try Google Colab's userdata
                if not self.api_key:
                    try:
                        from google.colab import userdata
                        self.api_key = userdata.get('OPENAI_API_KEY')
                    except ImportError:
                        pass
                
                if not self.api_key:
                    raise ModelInitializationError(
                        "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                        "or pass api_key parameter, or store in Google Colab secrets."
                    )
            
            # Initialize client
            self.client = openai.OpenAI(api_key=self.api_key)
            
            # Test authentication with a minimal request
            try:
                self.client.models.list()
                logger.info(f"OpenAI model {self.model_name} initialized successfully")
                
            except Exception as e:
                raise ModelInitializationError(f"OpenAI authentication failed: {e}")
            
            self.is_initialized = True
            
        except ImportError:
            raise ModelInitializationError(
                "OpenAI package not available. Install with: !pip install openai"
            )
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize OpenAI model: {e}")
    
    def _rate_limit(self) -> None:
        """Implement simple rate limiting."""
        if self.requests_per_minute > 0:
            min_interval = 60.0 / self.requests_per_minute
            elapsed = time.time() - self.last_request_time
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def predict(self, text: str) -> List[NERPrediction]:
        """
        Predict named entities using OpenAI API.
        
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
            self._rate_limit()
            
            # Create the API request
            messages = [
                {
                    "role": "system",
                    "content": f"{self.prompt}\n\nProcess this text: {text}"
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse response based on output format
            if self.output_format == "json":
                return self._parse_json_response(content, text)
            else:
                return self._parse_list_response(content, text)
                
        except Exception as e:
            logger.warning(f"Error in OpenAI prediction for {self.model_name}: {e}")
            raise ModelPredictionError(f"OpenAI prediction failed: {e}")
    
    def _parse_json_response(self, content: str, original_text: str) -> List[NERPrediction]:
        """Parse JSON-formatted response from OpenAI."""
        try:
            # Try to parse as JSON
            entities = json.loads(content)
            
            predictions = []
            for entity in entities:
                if isinstance(entity, dict) and 'name' in entity:
                    name = entity['name']
                    start = entity.get('start', 0)
                    end = entity.get('end', len(name))
                    
                    # Validate positions
                    if 0 <= start < end <= len(original_text):
                        predictions.append(NERPrediction(
                            start=start,
                            end=end,
                            entity_type='PERSON',
                            text=name
                        ))
            
            return predictions
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Fallback to list parsing
            return self._parse_list_response(content, original_text)
    
    def _parse_list_response(self, content: str, original_text: str) -> List[NERPrediction]:
        """Parse list-formatted response from OpenAI."""
        try:
            # Try to evaluate as Python list
            names = ast.literal_eval(content)
            if not isinstance(names, list):
                names = [names]
                
        except (ValueError, SyntaxError):
            # Fallback: extract from text using regex
            names = self._extract_names_from_text(content)
        
        # Find positions of names in original text
        return self._find_name_positions(original_text, names)
    
    def _extract_names_from_text(self, content: str) -> List[str]:
        """Extract names from response text using regex patterns."""
        # Look for quoted strings or comma-separated values
        patterns = [
            r'"([^"]+)"',  # Quoted strings
            r"'([^']+)'",  # Single quoted strings
            r'\[([^\]]+)\]',  # Content within brackets
        ]
        
        names = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                if len(matches) == 1 and ',' in matches[0]:
                    # Split comma-separated values
                    names.extend([name.strip().strip('"\'') for name in matches[0].split(',')])
                else:
                    names.extend(matches)
                break
        
        return [name for name in names if name.strip()]
    
    def _find_name_positions(self, text: str, names: List[str]) -> List[NERPrediction]:
        """Find character positions of names in text."""
        predictions = []
        start_index = 0
        
        for name in names:
            if not name:
                continue
                
            # Find the name in the text
            position = text.find(name, start_index)
            if position != -1:
                predictions.append(NERPrediction(
                    start=position,
                    end=position + len(name),
                    entity_type='PERSON',
                    text=name
                ))
                start_index = position + len(name)  # Prevent re-finding the same occurrence
            else:
                # If exact match not found, try case-insensitive
                position = text.lower().find(name.lower(), start_index)
                if position != -1:
                    actual_text = text[position:position + len(name)]
                    predictions.append(NERPrediction(
                        start=position,
                        end=position + len(name),
                        entity_type='PERSON',
                        text=actual_text
                    ))
                    start_index = position + len(name)
        
        return predictions
    
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including OpenAI-specific details."""
        info = super().get_model_info()
        info.update({
            'provider': 'OpenAI',
            'output_format': self.output_format,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'rate_limit': f"{self.requests_per_minute} requests/minute"
        })
        return info