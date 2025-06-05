"""
Common utility functions shared across the NER evaluation framework.
"""

import ast
import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def safe_eval_list(value: Any, error_context: str = "data") -> List:
    """
    Safely evaluate string representation of list or return empty list.
    
    This function handles various formats that might represent entity lists:
    - String representations of lists: "[(0, 5, 'PERSON')]"
    - Already parsed lists: [(0, 5, 'PERSON')]
    - Empty or invalid values: None, "", "[]"
    
    Args:
        value: Value to parse as a list
        error_context: Context for error messages (e.g., "data", "entity list")
        
    Returns:
        List representation of the value, or empty list if parsing fails
        
    Examples:
        >>> safe_eval_list("[(0, 5, 'PERSON')]")
        [(0, 5, 'PERSON')]
        >>> safe_eval_list([])
        []
        >>> safe_eval_list("invalid")
        []
    """
    if value is None or value == "" or value == "[]":
        return []
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, str):
        try:
            # Try literal_eval first (safer than eval)
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            else:
                logger.warning(f"Parsed {error_context} is not a list: {type(parsed)}")
                return []
        except (ValueError, SyntaxError):
            try:
                # Fallback to eval for edge cases
                parsed = eval(value)
                if isinstance(parsed, list):
                    return parsed
                else:
                    logger.warning(f"Evaluated {error_context} is not a list: {type(parsed)}")
                    return []
            except Exception as e:
                logger.warning(f"Failed to parse {error_context}: {e}")
                return []
    
    logger.warning(f"Unexpected {error_context} type: {type(value)}")
    return []


def normalize_entity_type(entity_type: str) -> str:
    """
    Normalize entity type to standard format.
    
    Args:
        entity_type: Entity type string (e.g., 'PERSON', 'PER', 'person')
        
    Returns:
        Normalized entity type ('PERSON' for person entities, original otherwise)
    """
    if entity_type.upper() in {'PERSON', 'PER'}:
        return 'PERSON'
    return entity_type.upper()


def _validate_entity_tuple(entity: Any) -> bool:
    """
    Internal helper to validate entity tuple format.
    
    Args:
        entity: Entity tuple to validate
        
    Returns:
        True if entity has format (start: int, end: int, type: str)
    """
    return (
        isinstance(entity, (tuple, list)) and
        len(entity) >= 3 and
        isinstance(entity[0], int) and
        isinstance(entity[1], int) and
        isinstance(entity[2], str) and
        entity[0] <= entity[1]
    )


def filter_person_entities(entities: List) -> List:
    """
    Filter entities to only include person-type entities.
    
    Args:
        entities: List of entity tuples
        
    Returns:
        List of person entities only
    """
    if not entities:
        return []
    
    person_entities = []
    for entity in entities:
        if _validate_entity_tuple(entity):
            entity_type = normalize_entity_type(entity[2])
            if entity_type == 'PERSON':
                # Return with normalized type
                person_entities.append((entity[0], entity[1], entity_type))
    
    return person_entities


# Removed setup_logging function as it's unused in the project